# type:ignore
from muFFT import FFT
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
import numpy as np

from plotting import create_interactive_u, update_interactive_u, create_interactive_powerspec, update_interactive_powerspec

# nu = 0.000625
# nu = 1/1600
nu = 1/800
velocity_amplitude = 0.001
dealias_or_not = True
forcing = True
# nu = 0.05
T = 10.0
# T = 1.0
dt = 0.001
N = 2**6
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N // num_processes
X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*2*pi/N
X /= 2  # smaller vortex
U = empty((3, Np, N, N))
U_hat = empty((3, N, Np, N//2+1), dtype=complex)
P = empty((Np, N, N))
P_hat = empty((N, Np, N//2+1), dtype=complex)
U_hat0 = empty((3, N, Np, N//2+1), dtype=complex)
U_hat1 = empty((3, N, Np, N//2+1), dtype=complex)
dU = empty((3, N, Np, N//2+1), dtype=complex)
Uc_hat = empty((N, Np, N//2+1), dtype=complex)
Uc_hatT = empty((Np, N, N//2+1), dtype=complex)
U_mpi = empty((num_processes, Np, Np, N//2+1), dtype=complex)
curl = empty((3, Np, N, N))
kx = fftfreq(N, 1./N)
kz = kx[:(N//2+1)].copy()
kz[-1] *= -1
K = array(meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'), dtype=int)
K2 = sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
kmax_dealias = 2./3.*(N//2+1)
print("dealias, min, max", kmax_dealias,
      np.min(K), np.max(K))
dealias = array((abs(K[0]) < kmax_dealias)*(abs(K[1]) <
                kmax_dealias)*(abs(K[2]) < kmax_dealias), dtype=bool)
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]


def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT .shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u


def fftn_mpi(u, fu):
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np, N//2+1), 1)
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu


def Cross(a, b, c):
    c[0] = fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    c[1] = fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    c[2] = fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])
    return c


def Curl(a, c):
    c[2] = ifftn_mpi(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    c[1] = ifftn_mpi(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[0] = ifftn_mpi(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c


def computeRHS(dU, rk):
    if rk > 0:
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
    curl[:] = Curl(U_hat, curl)
    dU = Cross(U, curl, dU)
    if dealias_or_not:
        dU *= dealias
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)
    dU -= P_hat*K
    dU -= nu*K2*U_hat
    if forcing:
        mask = (abs(K) == 1) | (abs(K) == 2)
        dU[mask] = 0
    return dU


n = N
nb_grid_pts = (n, n, n)
physical_sizes = (1, 1, 1)
nx, ny, nz = nb_grid_pts
lx, ly, lz = physical_sizes
fft_grid = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)
# velocity_amplitude = 0.07


def random_field():
    # modification: seed the rng to 42
    # CODE FROM https://pastewka.github.io/SpectralMethods/_project/milestone03.html START
    k = (2 * np.pi * fft_grid.fftfreq.T / np.array(physical_sizes)).T
    k_eq_0_mask = (k.T == np.zeros(3, dtype=int)).T.all(axis=0)
    k_sq = np.sum(k ** 2, axis=0)
    # Fourier space velocity field
    random_field = np.zeros((3,) + fft_grid.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng(42)
    random_field.real = rng.standard_normal(random_field.shape)
    random_field.imag = rng.standard_normal(random_field.shape)
    # Initial velocity field should decay as k^(-5/3) for the Kolmogorov spectrum
    fac = np.zeros_like(k_sq)
    # Avoid division by zero
    fac[np.logical_not(k_eq_0_mask)] = velocity_amplitude * \
        k_sq[np.logical_not(k_eq_0_mask)] ** (-5 / 6)
    random_field *= fac
    # CODE FROM https://pastewka.github.io/SpectralMethods/_project/milestone03.html END
    k_orig = k.copy()

    print("k_orig_shape", k_orig.shape)
    k_small_mask = (np.abs(k.T) <= 0.2*np.ones(3, dtype=int)).T.all(axis=0)
    print("small mask", k_small_mask.shape)

    print("divergence before:", np.abs(np.mean(random_field * k)))
    u_f_init = fft_grid.fourier_space_field('u_hat', (3,))
    # make the random_field divergence free
    k_k_tensor_prod = np.einsum('i...,j...->ij...', k, k)
    k_tensor_k_over_k_sq = k_k_tensor_prod / np.where(k_sq == 0, 1, k_sq)
    id = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
    project_div_free = id - k_tensor_k_over_k_sq
    u_f_init.p = np.einsum('ij...,j...->i...', project_div_free, random_field)
    print("divergence after:", np.abs(np.mean(u_f_init.p * k)))

    # find initial real-space u from u_f_init.p
    u_r_init = fft_grid.real_space_field('u', (3,))
    fft_grid.ifft(u_f_init, u_r_init)
    u = u_r_init.p
    print("kk", K.shape, k.shape)
    u_compat = np.swapaxes(u, 1, 3)
    print("uu", u_compat.shape, U.shape)
    return u_compat


def random_field_2():
    q = (2 * np.pi * fft_grid.fftfreq.T / (np.array(physical_sizes)/N)).T
    zero_wavevector = (q.T == np.zeros(3, dtype=int)).T.all(axis=0)
    q_sq = np.sum(q ** 2, axis=0)
    random_field = np.zeros((3,) + fft_grid.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    random_field.real = rng.standard_normal(random_field.shape)
    random_field.imag = rng.standard_normal(random_field.shape)
    fac = np.zeros_like(q_sq)
    fac[np.logical_not(zero_wavevector)] = velocity_amplitude * \
        q_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)
    random_field *= fac
    qxq = np.zeros((3, *q.shape))
    for i in range(3):
        for j in range(3):
            qxq[i][j] = q[i]*q[j]
    u_hat = fft_grid.fourier_space_field('u_hat', (3,))
    u = fft_grid.real_space_field('u', (3,))
    # build compatible 3x3 identity matrix
    ident = np.zeros_like(qxq)
    for i in range(3):
        ident[i][i] = np.ones(qxq[0][0].shape)
    # Calculate "Transform"-Matrix
    incomp_matrix = ident - 1/q_sq*qxq
    # Fix division by zero
    incomp_matrix[:, :, 0, 0, 0] = 0
    for i in range(3):
        for j in range(3):
            u_hat.p[i] += incomp_matrix[i][j]*random_field[j]

    fft_grid.ifft(u_hat, u)
    return u.p


u_compat = random_field()
# U[0] = sin(X[0])*cos(X[1])*cos(X[2])
# U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
# U[2] = 0

U[0] = u_compat[0]
U[1] = u_compat[1]
U[2] = u_compat[2]


for i in range(3):
    U_hat[i] = fftn_mpi(U[i], U_hat[i])

t = 0.0
tstep = 0

# img, fig, ax = create_interactive_u(U)
fig2, ax2, sct2 = create_interactive_powerspec(K, K, N)


def v_avg(): return mean(sqrt(U[0]**2 + U[1]**2 + U[2]**2))


v_0 = v_avg()
def v_ana(): return mean(v_0 * exp(-2.0*nu*t))


while t < T - 1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = computeRHS(dU, rk)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1[:] += a[rk]*dt*dU
    U_hat[:] = U_hat1[:]
    for i in range(3):
        U[i] = ifftn_mpi(U_hat[i], U[i])
    # update graphics and print
    # update_interactive_u(img, fig, ax, U)
    update_interactive_powerspec(U_hat, K, ax2, fig2, N)
    k = comm.reduce(0.5*sum(U*U)*(1./N)**3)
    print("t={:.4f}, k={}, (v_avg-v_ana)/v_avg={}".format(
        t, k, (v_avg()-v_ana())/v_avg()), end="\r")

# if rank == 0:
#     assert round(k - 0.124953117517, 7) == 0
#     print("done")
