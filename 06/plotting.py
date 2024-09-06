
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.stats import binned_statistic


def create_interactive_u(u):
    fig, ax = plt.subplots()
    img = ax.imshow(np.linalg.norm(u[:, :, :, 0], axis=0), animated=True)
    plt.show(block=False)
    return img, fig, ax


def update_interactive_u(img, fig, ax, u):
    img.set_data(np.linalg.norm(u[:, :, :, 0], axis=0))
    # img.set_data(np.random.random((n, n)))
    ax.draw_artist(img)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()


def avg_speed(u) -> float: return np.mean(np.sqrt(u[0]**2 + u[1]**2 + u[2]**2))


def print_and_record_v(v_avg, xs, t, u):
    if MPI.COMM_WORLD.rank == 0:
        xs += [t]
        v_avg += [avg_speed(u)]
        print(
            "t={:.4f}".format(t),
            "\t v_avg={:.10f}".format(v_avg[-1]),
            end="\r"
        )


def create_interactive_powerspec(u_f_cur, k, n):
    k_mags = np.linalg.norm(k, axis=0)
    energy_spectrum = 0.5 * np.abs(u_f_cur)**2
    e = np.sum(energy_spectrum, axis=0)

    x_min = max(np.min(k_mags), 1/n)
    x_max = np.max(k_mags)
    xs = np.linspace(x_min, x_max, 100)
    bin_means, bin_edges, _ = binned_statistic(
        x=k_mags.flatten(),
        values=e.flatten(),
        bins=np.logspace(np.log10(x_min), np.log10(x_max), 50)
    )
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    fig, ax = plt.subplots()
    sct = ax.scatter(k_mags, e, s=0.1)
    plt.show(block=False)
    return fig, ax, sct


def update_interactive_powerspec(u_f_cur, k, ax, fig, n):
    k_mags = np.linalg.norm(k, axis=0)
    e = np.sum(0.5 * np.abs(u_f_cur)**2, axis=0)

    x_min = max(np.min(k_mags), 1/n)
    x_max = np.max(k_mags)
    xs = np.linspace(x_min, x_max, 100)
    bin_means, bin_edges, _ = binned_statistic(
        x=k_mags.flatten(),
        values=e.flatten(),
        bins=np.logspace(np.log10(x_min), np.log10(x_max), 50)
    )
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    ax.clear()
    ax.loglog(bin_centers, bin_means,  "-r")
    ax.loglog(xs, xs**(-10/3),  "-r")
    ax.scatter(k_mags, e, s=0.1)
    # ax.set_ylim(10**-9, 1)
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_binned_spec(k, e, n):
    k = k*k
    x_min = max(np.min(k), 1/n)
    x_max = np.max(k)
    xs = np.linspace(x_min, x_max, 100)
    bin_means, bin_edges, _ = binned_statistic(
        x=k.flatten(),
        values=e.flatten(),
        bins=np.logspace(np.log10(x_min), np.log10(x_max), 20)
    )
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    plt.scatter(k, e, s=0.1)
    plt.plot(bin_centers, bin_means, "r-")
    plt.plot(xs, xs**(-5/3), "g-")
    # plt.ylim(10**-8, 0.1)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def plot_v_avg_evolution(xs, v_avg,  filename):
    # styling and setup
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(xs, v_avg, "-r")
    ax.set_title(
        r"Time Evolution of the Average Velocity")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Average Velocity $|\vec{\mathbf{v}}|_{avg}$")
    # fig.text(0.99, 0.01, info, horizontalalignment='right', fontsize="8")
    fig.savefig(filename, dpi=400)


# def plot_power_spectrum(u_f, name):
#     # k_mags = np.linalg.norm(k, axis=0, ord=2.0)**2
#     k_mags = np.linalg.norm(k, axis=0)
#     # e = 0.5*np.square(np.linalg.norm(u_f, axis=0))

#     energy_spectrum = 0.5 * np.abs(u_f)**2
#     e = np.sum(energy_spectrum, axis=0)

#     x_min = max(np.min(k_mags), 1/nb_grid_pts[0])
#     x_max = np.max(k_mags)
#     xs = np.linspace(x_min, x_max, 100)
#     bin_means, bin_edges, _ = binned_statistic(
#         x=k_mags.flatten(),
#         values=e.flatten(),
#         bins=np.logspace(np.log10(x_min), np.log10(x_max), 50)
#     )
#     bin_width = (bin_edges[1] - bin_edges[0])
#     bin_centers = bin_edges[1:] - bin_width/2

#     fig, ax = plt.subplots(1, 1, figsize=(6, 8))
#     ax.loglog(bin_centers, bin_means,  "-r")

#     ax.plot(xs, xs**(-5/3), "-g")
#     ax.plot(xs, xs**(-5/6), "-b")
#     ax.scatter(k_mags, e, s=0.1)
#     fig.savefig(name+".png")


# info = (
#     r"$\nu="+str(nu)
#     + r", \Delta t="+str(round(dt, 5))
#     + r", (x,y)\in[0; 1]^3$ on a $"
#     + str(nx)+r"\times "
#     + str(ny)+r"\times "
#     + str(nz)
#     + r"$ Grid, "
#     + r"$\\\vec{\mathbf{v}}_0\sim\mathcal{N}(0,1)$ modified such that $\nabla\cdot\vec{\mathbf{v}}_0=0, E_v(q)\propto q^{-\frac{5}{3}}$"
# )


def plot_u(u, t, filename, fft, physical_sizes, slice=0):
    print("Plotting ...")
    print(u.shape)
    lx, ly, lz = physical_sizes
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
    fig.text(0.99, 0.03, "", horizontalalignment='right', fontsize="8")
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
