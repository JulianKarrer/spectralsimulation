# type:ignore
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
nu = 0.000625
T = 0.1
# T = 1.0
dt = 0.01
N = 2**7
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N // num_processes
X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*2*pi/N
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
    dU *= dealias
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)
    dU -= P_hat*K
    dU -= nu*K2*U_hat
    return dU


U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0


for i in range(3):
    U_hat[i] = fftn_mpi(U[i], U_hat[i])
t = 0.0
tstep = 0
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
k = comm.reduce(0.5*sum(U*U)*(1./N)**3)
if rank == 0:
    assert round(k - 0.124953117517, 7) == 0
    print("done")
