# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from mpi4py import MPI
import time


# mpirun -np 4 python3 vortex.py
# 32x32x32 is fine
n = 256
nb_grid_pts = (n, n, 2)
physical_sizes = (2*np.pi, 2*np.pi, 2*np.pi)
nx, ny, nz = nb_grid_pts
lx, ly, lz = physical_sizes
fft = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)

# # test if MPI works, taken from muspectre docs
# if MPI.COMM_WORLD.rank == 0:
#     print('  Rank   Size          Domain       Subdomain        Location')
#     print('  ----   ----          ------       ---------        --------')

# MPI.COMM_WORLD.Barrier()
# print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(fft.nb_domain_grid_pts):>15} '
#       f'{str(fft.nb_subdomain_grid_pts):>15} {str(fft.subdomain_locations):>15}')

# SETTINGS
nu = 0.05
lam = 0.015
t_end = 10

# maximum timestep by CFL condition
v_max = 1  # no velocity > 1 will be reached due to initial conditions and v decaying
dt_max = min([lx/nx, ly/ny, lz/nz])/v_max
dt = lam * dt_max
print(dt_max, dt, lam)


info = (
    r"$\nu="+str(nu)
    + r", \Delta t="+str(round(dt, 5))
    + r", (x,y)\in[0; 2\pi ]^2$ on a $"
    + str(nx)+r"\times "
    + str(ny)+r"\times "
    + str(nz)
      + r"$ Grid, $u_0 =\sin(x)\cos(y), v_0 =-\cos(x)\sin(y)$")


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


def rk2(f, _t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(0, y)
    k2 = f(0, y + dt / 2 * k1)
    return dt * k2


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


def plot_v_avg_evolution(xs, v_avg, v_ana, filename):
    # styling and setup
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
    fig.savefig(filename, dpi=400)


def print_and_record_v(v_avg, v_ana, xs, t, u):
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


def sim_explicit_euler(u=analytic(0)):
    t = 0
    xs: list[float] = []
    v_avg: list[float] = []
    v_ana: list[float] = []
    while t <= t_end:
        t += dt
        # update u
        u += du_dt(t, u) * dt
        # print current state and record velocities
        print_and_record_v(v_avg, v_ana, xs, t, u)
    return xs, v_avg, v_ana


def sim_rk4(u=analytic(0)):
    t = 0
    xs: list[float] = []
    v_avg: list[float] = []
    v_ana: list[float] = []
    while t <= t_end:
        t += dt
        # update u
        u += rk4(du_dt, t, u, dt)
        # print current state and record velocities
        print_and_record_v(v_avg, v_ana, xs, t, u)
    return xs, v_avg, v_ana


def sim_rk2(u=analytic(0)):
    t = 0
    xs: list[float] = []
    v_avg: list[float] = []
    v_ana: list[float] = []
    while t <= t_end:
        t += dt
        # update u
        u += rk2(du_dt, t, u, dt)
        # print current state and record velocities
        print_and_record_v(v_avg, v_ana, xs, t, u)
    return xs, v_avg, v_ana


def sim_adam_bash_2(u=analytic(0)):
    t = 0
    xs: list[float] = []
    v_avg: list[float] = []
    v_ana: list[float] = []

    # use runge kutta for the initial step
    du = rk4(du_dt, 0, u, dt)
    prev_du_dt = du/dt
    u += du
    t += dt

    while t <= t_end:
        t += dt
        # update u
        du = du_dt(0, u)
        u += dt/2*(3*du - prev_du_dt)
        prev_du_dt = du
        # print current state and record velocities
        print_and_record_v(v_avg, v_ana, xs, t, u)
    return xs, v_avg, v_ana


def bench():
    start = time.perf_counter()
    xs_ab2, v_avg_ab2, v_ana_ab2 = sim_adam_bash_2()
    time_ab2 = time.perf_counter()-start
    if MPI.COMM_WORLD.rank == 0:
        print("\nab2:", time_ab2)

    start = time.perf_counter()
    xs_exe, v_avg_exe, v_ana_exe = sim_explicit_euler()
    time_exe = time.perf_counter()-start
    if MPI.COMM_WORLD.rank == 0:
        print("\nexe:", time_exe)

    start = time.perf_counter()
    xs_rk2, v_avg_rk2, v_ana_rk2 = sim_rk2()
    time_rk2 = time.perf_counter()-start
    if MPI.COMM_WORLD.rank == 0:
        int(nz/2)
    xs_rk4, v_avg_rk4, v_ana_rk4 = sim_rk4()
    time_rk4 = time.perf_counter()-start
    if MPI.COMM_WORLD.rank == 0:
        print("\nrk4:", time_rk4)

    if MPI.COMM_WORLD.rank == 0:
        # first plot: accuracy
        plt.style.use('default')
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(xs_ab2, [abs(s-v) for v, s in zip(v_avg_ab2, v_ana_ab2)],
                label="Adams-Bashforth 2")
        ax.plot(xs_exe, [abs(s-v) for v, s in zip(v_avg_exe, v_ana_exe)],
                label="Explicit Euler")
        ax.plot(xs_rk2, [abs(s-v) for v, s in zip(v_avg_rk2, v_ana_rk2)],
                label="Explicit Midpoint (RK2)")
        ax.plot(xs_rk4, [abs(s-v) for v, s in zip(v_avg_rk4, v_ana_rk4)],
                label="Runge Kutta 4")
        ax.legend()
        ax.set_title(
            r"Time Evolution of Average Velocity Magnitude Error")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(
            r"Average Velocity Magnitude Error $v_{ana} - |\vec{\mathbf{v}}|_{avg}$")
        ax.set_yscale('log')
        fig.text(0.99, 0.01, info, horizontalalignment='right', fontsize="8")
        fig.savefig("bench_accuracy.png", dpi=300)

        # second plot: accuracy vs computation time
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.scatter(
            x=time_ab2, y=v_ana_ab2[-1] - v_avg_ab2[-1], label="Adams-Bashforth 2")
        ax2.scatter(
            x=time_exe, y=v_ana_exe[-1] - v_avg_exe[-1], label="Explicit Euler")
        ax2.scatter(
            x=time_rk2, y=v_ana_rk2[-1] - v_avg_rk2[-1], label="Explicit Midpoint (RK2)")
        ax2.scatter(
            x=time_rk4, y=v_ana_rk4[-1] - v_avg_rk4[-1], label="Runge Kutta 4")
        ax2.legend()
        ax2.set_title(
            r"Simulation Accuracy vs. Computation Time")
        ax2.set_xlabel(r"Computation Time (s)")
        ax2.set_ylabel(
            r"At $t="+str(t_end)+r"$: Average Velocity Magnitude Error $v_{ana}(t_{end}) - |\vec{\mathbf{v}}|_{avg}(t_{end})$")
        fig2.text(0.99, 0.01, info, horizontalalignment='right', fontsize="8")
        fig2.savefig("bench_time_vs_accuracy.png", dpi=300)


if __name__ == "__main__":
    # # benchmark explicit methods
    # bench()

    # plot the initial velocities(analytic solution at t=0)
    if MPI.COMM_WORLD.rank == 0:
        plot_u(u=analytic(0), t=0, filename="vortex_initial.png")

    u = analytic(0)
    xs, v_avg, v_ana = sim_adam_bash_2(u)

    if MPI.COMM_WORLD.rank == 0:
        print()
        # show final velocity field
        plot_u(u=u, t=t_end, filename="vortex_final.png")
        # show time evolution of average speed
        plot_v_avg_evolution(xs, v_avg, v_ana, "average_v.png")
