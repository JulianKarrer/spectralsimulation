# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.stats import binned_statistic
from plotting import create_interactive_u, update_interactive_u, create_interactive_powerspec, update_interactive_powerspec, print_and_record_v, plot_binned_spec, plot_u, plot_v_avg_evolution

# mpirun -np 4 python3 vortex.py
# 32x32x32 is fine

# SETTINGS
# nu = 1/1600
nu = 0.1
dt = 0.01
t_end = 10
velocity_amplitude = 0.001
# velocity_amplitude = 0.0003
n = 2**6
forcing = True
dealias = False
physical_sizes = (np.pi, np.pi, np.pi)

nb_grid_pts = (n, n, n)
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

k_int = fft.ifftfreq
k_dealias_thresh = 2./3.*(n//2+1)
dealias_correction = np.array(
    (np.abs(k_int[0]) < k_dealias_thresh) *
    (np.abs(k_int[1]) < k_dealias_thresh) *
    (np.abs(k_int[2]) < k_dealias_thresh),
    dtype=bool)

# print("k_orig_shape", k_orig.shape)
# k_small_mask = (np.abs(k.T) <= 0.2*np.ones(3, dtype=int)).T.all(axis=0)
# print("small mask", k_small_mask.shape)


print("divergence before:", np.abs(np.mean(random_field * k)))
u_f_init = fft.fourier_space_field('u_hat', (3,))
# make the random_field divergence free
k_k_tensor_prod = np.einsum('i...,j...->ij...', k, k)
k_tensor_k_over_k_sq = k_k_tensor_prod / np.where(k_sq == 0, 1, k_sq)
id = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
project_div_free = id - k_tensor_k_over_k_sq
u_f_init.p = np.einsum('ij...,j...->i...', project_div_free, random_field)
print("divergence after:", np.abs(np.mean(u_f_init.p * k)))


# find initial real-space u from u_f_init.p
u_f_cur = u_f_init.p
u_f_orig = u_f_init.p.copy()
u_r_init = fft.real_space_field('u', (3,))
fft.ifft(u_f_init, u_r_init)
u = u_r_init.p

# oooorrrrr do a taylor green vortex lol
# x = np.array([fft.coords[0]*lx, fft.coords[1]*ly, fft.coords[2]*lz])
# u = np.zeros(fft.coords.shape)
# u[0] = np.sin(x[0])*np.cos(x[1])
# u[1] = -np.cos(x[0])*np.sin(x[1])


# plot the spectrum


def power_spectrum(u_f_cur, k):
    k_mags = np.linalg.norm(k, axis=0)
    u_mags = np.linalg.norm(u_f_cur, axis=0)
    energy_spectrum = 0.5 * u_mags**2
    return k_mags, energy_spectrum


_k, _e = power_spectrum(u_f_cur, k_int)
plot_binned_spec(_k, _e, n)

# precompute k/(k**2)
k_2 = k_sq
k_over_k_2 = np.divide(k, np.where(k_2 == 0, 1.0, k_2))


def curl(u_f, temp_f, curl_r):
    (k_x, k_y, k_z) = (fft.fftfreq[0], fft.fftfreq[1], fft.fftfreq[2])
    (u_x, u_y, u_z) = (u_f.p[0], u_f.p[1], u_f.p[2])
    dyuz = 1j * k_y * u_z
    dzuy = 1j * k_z * u_y
    dzux = 1j * k_z * u_x
    dxuz = 1j * k_x * u_z
    dxuy = 1j * k_x * u_y
    dyux = 1j * k_y * u_x

    temp_f.p[0] = (dyuz - dzuy)
    temp_f.p[1] = (dzux - dxuz)
    temp_f.p[2] = (dxuy - dyux)
    fft.ifft(temp_f, curl_r)


def cross(x, y, temp_r, res_f):
    temp_r.p[0] = x.p[1]*y.p[2] - x.p[2]*y.p[1]
    temp_r.p[1] = x.p[2]*y.p[0] - x.p[0]*y.p[2]
    temp_r.p[2] = x.p[0]*y.p[1] - x.p[1]*y.p[0]
    fft.fft(temp_r, res_f)
    res_f.p *= fft.normalisation


def du_dt(_t, u):
    # initialize fields to be used in calculations by du_dt
    u_r = fft.real_space_field('u', (3,))
    u_f = fft.fourier_space_field('u_hat', (3,))
    du_r = fft.real_space_field('du', (3,))
    du_f = fft.fourier_space_field('du_hat', (3,))
    p_f = fft.fourier_space_field('p_hat', 3)
    curl_r = fft.real_space_field('curl_r', (3,))
    temp_f = fft.fourier_space_field('temp_f', (3,))
    temp_r = fft.real_space_field('temp_r', (3,))

    u_r.p = u.copy()
    # transform velocity field to fourier space
    fft.fft(u_r, u_f)
    global u_f_cur
    u_f_cur = u_f.p
    u_f.p *= fft.normalisation
    # convective term
    curl(u_f, temp_f, curl_r)
    cross(u_r, curl_r, temp_r, du_f)

    # TODO dealias
    if dealias:
        du_f.p *= dealias_correction

    # pressure poisson equation
    p_f.p = du_f.p*k_over_k_2
    # pressure gradient
    du_f.p -= p_f.p*k
    # viscous term
    du_f.p -= nu*k_2*u_f.p
    # freeze lowest wavenumbers by setting du/dt to zero
    driven = du_f.p.copy()
    if forcing:
        # print("driven shape", driven.shape)
        # driven[:, :freeze_k, :freeze_k, :freeze_k] = 0
        mask = (np.abs(k_int) == 1) | (np.abs(k_int) == 2)
        driven[mask] = 0
    # transform back du_dt to real space
    fft.ifft(driven, du_r)
    return du_r.p


def rk4(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def avg_speed(u) -> float: return np.mean(np.sqrt(u[0]**2 + u[1]**2 + u[2]**2))


if __name__ == "__main__":
    t = 0
    xs: list[float] = []
    v_avg: list[float] = []

    # use runge kutta for the initial step
    du = rk4(du_dt, 0, u, dt)
    prev_du_dt = du/dt
    u += du
    t += dt

    img, fig, ax = create_interactive_u(u)

    # plot_u(u, t, "start.png", 0)
    # plot_power_spectrum(u_f_cur, "spec_start")
    fig_2, ax_2, sct_2 = create_interactive_powerspec(u_f_cur, k, n)
    i = 0
    while t <= t_end:
        t += dt
        # update u
        # du = du_dt(0, u)
        # u += (dt/2*(3*du - prev_du_dt))
        # prev_du_dt = du
        du = rk4(du_dt, 0, u, dt)
        u += du*dt

        # print current state and record velocities
        print_and_record_v(v_avg, xs, t, u)
        # update image
        if i % 10 == 0:
            pass
            update_interactive_u(img, fig, ax, u)
            update_interactive_powerspec(u_f_cur, k, ax_2, fig_2, n)
        i += 1

    _k, _e = power_spectrum(u_f_cur, k_int)
    plot_binned_spec(_k, _e, n)
    # plot_power_spectrum(u_f_cur, "spec_end")
    plot_u(u, t, "end.png", fft, physical_sizes, 0)
    plot_v_avg_evolution(xs, v_avg, "average_v.png")
    plt.show()
