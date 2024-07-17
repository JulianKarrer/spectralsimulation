# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.stats import binned_statistic

# mpirun -np 4 python3 vortex.py
# 32x32x32 is fine
n = 128
nb_grid_pts = (n, n, n)
physical_sizes = (1, 1, 1)
nx, ny, nz = nb_grid_pts
lx, ly, lz = physical_sizes
fft = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)

# test if MPI works, taken from muspectre docs
if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')

MPI.COMM_WORLD.Barrier()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(fft.nb_domain_grid_pts):>15} '
      f'{str(fft.nb_subdomain_grid_pts):>15} {str(fft.subdomain_locations):>15}')

# SETTINGS
nu = 1/1600
dt = 0.001
t_end = 1
velocity_amplitude = 1


info = (
    r"$\nu="+str(nu)
    + r", \Delta t="+str(round(dt, 5))
    + r", (x,y)\in[0; 1]^3$ on a $"
    + str(nx)+r"\times "
    + str(ny)+r"\times "
    + str(nz)
    + r"$ Grid, "
    + r"$\\\vec{\mathbf{v}}_0\sim\mathcal{N}(0,1)$ modified such that $\nabla\cdot\vec{\mathbf{v}}_0=0, E_v(q)\propto q^{-\frac{5}{3}}$"
)


# modification: seed the rng to 42
# CODE FROM https://pastewka.github.io/SpectralMethods/_project/milestone03.html START
k = (2 * np.pi * fft.fftfreq.T / np.array(physical_sizes)).T
k_eq_0_mask = (k.T == np.zeros(3, dtype=int)).T.all(axis=0)
k_sq = np.sum(k ** 2, axis=0)
# Fourier space velocity field
random_field = np.zeros((3,) + fft.nb_fourier_grid_pts, dtype=complex)
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


def plot_u(u, t, filename, slice=0):
    print("Plotting ...")
    print(u.shape)
    x = np.array([fft.coords[0]*lx, fft.coords[1]*ly, fft.coords[2]*lz])
    X, Y = np.meshgrid(x[0, :, 0, 0], x[1, 0, :, 0])
    U, V = u[1, :, :, slice], u[0, :, :, slice]
    C = np.linalg.norm(u[:, :, :, slice], axis=0)

    # styling and setup
    plt.style.use('dark_background')
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    # plot description
    ax.set_title(r"Stream Plot of Turbulent Flow at $t=" +
                 "{:.1f}".format(t)+"$")
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    fig.text(0.99, 0.03, info, horizontalalignment='right', fontsize="8")
    # plot it
    _strm = ax.streamplot(
        X, Y, U, V, linewidth=0.4, color=C, cmap='Spectral_r', density=9)
    # create colourbar
    sm = plt.cm.ScalarMappable(
        cmap='Spectral_r', norm=plt.Normalize(vmin=C.min(), vmax=C.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, orientation="horizontal", pad=0.1, ax=ax)
    cbar.set_label(r'Velocity Magnitude $|\vec{\mathbf{v}}|$')
    # make axis pretty
    ax.set_aspect('equal')
    ax.margins(0)
    # save to file
    fig.savefig(filename, dpi=200)
    print("Done Plotting!")


def plot_power_spectrum(u):
    k_mags = np.linalg.norm(k, axis=0)**2
    u_sq = np.linalg.norm(u, axis=0)**2
    e = u_sq*0.5

    x_min = max(np.min(k_mags), 1/nb_grid_pts[0])
    x_max = np.max(k_mags)
    xs = np.linspace(x_min, x_max, 100)
    bin_means, bin_edges, _ = binned_statistic(
        x=k_mags.flatten(),
        values=e.flatten(),
        bins=np.logspace(np.log10(x_min), np.log10(x_max), 50)
    )
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    plt.loglog(bin_centers, bin_means,  "-r")

    plt.plot(xs, xs**(-5/3), "-g")
    plt.show()


# define fields
u_r = fft.real_space_field('u', (3,))
u_f = fft.fourier_space_field('u_hat', (3,))


print("divergence before:", np.abs(np.mean(random_field * k)))

# make the random_field divergence free
k_k_tensor_prod = np.einsum('i...,j...->ij...', k, k)
k_tensor_k_over_k_sq = k_k_tensor_prod / np.where(k_sq == 0, 1, k_sq)
id = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
project_div_free = id - k_tensor_k_over_k_sq
u_f.p = np.einsum('ij...,j...->i...', project_div_free, random_field)

print("divergence after:", np.abs(np.mean(u_f.p * k)))

fft.ifft(u_f, u_r)

if MPI.COMM_WORLD.rank == 0:
    plot_power_spectrum(u_f.p)
    # plot_u(u_r.p, 0, "rand.png", 0)
