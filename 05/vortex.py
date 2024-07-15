# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from mpi4py import MPI


# mpirun -np 4 python3 vortex.py
# 32x32x32 is fine
n = 256
nb_grid_pts = (n, n, 2)
physical_sizes = (2*np.pi, 2*np.pi, 2*np.pi)
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
nu = 0.05
dt = 0.001
t_end = 30

info = r"$\nu="+str(nu)+r", \Delta t="+str(dt) + \
    r", (x,y)\in[0; 2\pi ]^2, u_0 =\sin(x)\cos(y), v_0 =-\cos(x)\sin(y)$"


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


# precompute k/(k**2)
k = (2 * np.pi * fft.ifftfreq.T / np.array(physical_sizes)).T  # from muFFT example
k_2 = np.square(np.linalg.norm(k, axis=0))
k_over_k_2 = np.divide(k, np.where(k_2 == 0, 1.0, k_2))


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
    u_f.p *= fft.normalisation
    # convective term
    curl(u_f, temp_f, curl_r)
    cross(u_r, curl_r, temp_r, du_f)
    # pressure poisson equation
    p_f.p = du_f.p*k_over_k_2
    # pressure gradient
    du_f.p -= p_f.p*k
    # viscous term
    du_f.p -= nu*k_2*u_f.p
    # transform back du_dt to real space
    fft.ifft(du_f, du_r)
    return du_r.p.copy()


def rk4(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Implements the fourth-order Runge-Kutta method for numerical integration
    of multidimensional fields.

    Parameters
    ----------
    f : function
        The function to be integrated. It should take two arguments: time t
        and field y.
    t : float
        The current time.
    y : array_like
        The current value of the field.
    dt : float
        The time step for the integration.

    Returns
    -------
    dy : np.ndarray
        The increment of the field required to obtain the value at t + dt.
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def analytic(t: float):
    x = np.array([fft.coords[0]*lx, fft.coords[1]*ly, fft.coords[2]*lz])
    sol = np.zeros(fft.coords.shape)
    F_t = np.exp(-2.0*nu*t)
    sol[0] = np.sin(x[0])*np.cos(x[1])*F_t
    sol[1] = -np.cos(x[0])*np.sin(x[1])*F_t
    return sol


def average_speed(u) -> float:
    x, y, z = u[0], u[1], u[2]
    return np.mean(np.sqrt(x**2 + y**2 + z**2))


def plot_u(u, t, filename):
    x = np.array([fft.coords[0]*lx, fft.coords[1]*ly, fft.coords[2]*lz])
    X, Y = np.meshgrid(x[0, :, 0], x[1, 0, :])
    U, V = u[1, :, :, 0], u[0, :, :, 0]
    C = np.linalg.norm(u[:, :, :, 0], axis=0)

    # styling and setup
    plt.style.use('dark_background')
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    # plot description
    ax.set_title(r"Stream Plot of Taylor-Green Vortex at $t=" +
                 "{:.1f}".format(t)+"$")
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    fig.text(0.99, 0.01, info, horizontalalignment='right', fontsize="8")
    # plot it
    _strm = ax.streamplot(
        X, Y, U, V, linewidth=0.5, color=C, cmap='Spectral_r', density=5)
    # create colourbar
    sm = plt.cm.ScalarMappable(
        cmap='Spectral_r', norm=plt.Normalize(vmin=C.min(), vmax=C.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, orientation="horizontal", pad=0.1, ax=ax)
    cbar.set_label(r'Velocity Magnitude $|\vec{\mathbf{v}}|$')
    # make axis pretty
    ax.set_aspect('equal')
    ax.set_adjustable('datalim')
    ax.margins(0)
    # save to file
    fig.savefig(filename, dpi=400)


if __name__ == "__main__":
    # start the simulation
    t = 0
    xs = []
    v_avg = []
    v_ana = []

    # plot the initial velocities (analytic solution at t=0)
    if MPI.COMM_WORLD.rank == 0:
        plot_u(analytic(0), t, "vortex_initial.png")

    u = analytic(0)
    while t <= t_end:
        t += dt
        # update u
        # u += rk4(du_dt, t, u, dt)
        u += du_dt(t, u) * dt

        # print current state to stdout
        if MPI.COMM_WORLD.rank == 0:
            xs += [t]
            v_avg += [average_speed(u)]
            v_ana += [average_speed(analytic(t))]
            print(
                "t={:.4f}".format(t),
                "\t v_avg={:.10f}".format(v_avg[-1]),
                "\t v_ana={:.10f}".format(v_ana[-1]),
                "\t v_avg-v_ana={:.10f}".format(v_avg[-1]-v_ana[-1]),
                end="\r"
            )

    if MPI.COMM_WORLD.rank == 0:
        print()

        # show final velocity field
        plot_u(u, t, "vortex_final.png")

        # show time evolution of average speed
        plt.style.use('default')
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(xs, v_avg, "-r", label="measured velocities")
        ax.plot(xs, v_ana, ":b", label="analytic solution")
        ax.legend()
        ax.set_title(
            r"Time Evolution of the Average Velocity compared to Analytic Solution")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Average Velocity $|\vec{\mathbf{v}}|_{avg}$")
        fig.text(0.99, 0.01, info, horizontalalignment='right', fontsize="8")
        fig.savefig("average_v.png", dpi=400)
