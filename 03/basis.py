# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from matplotlib import cm
from mpi4py import MPI
from itertools import product
plt.rcParams['text.usetex'] = True

N = 100
x, y = nb_grid_pts = (N, N)
nx, ny = nb_grid_pts
fft = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)

qs = fft.real_space_field('qs')
qs.p = np.zeros(qs.p.shape)
qs_hat = fft.fourier_space_field('qs hat')


def update_qs(freq: int):
    fft.fft(qs, qs_hat)
    qs_hat.p *= fft.normalisation
    qs_hat.p[freq, 0] = 1
    qs_hat.p[0, freq] = 1
    fft.ifft(qs_hat, qs)
    return qs.p.T.copy()


n = 3
x = np.arange(x)/N
y = np.arange(y)/N
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, n + 1):
    Z = update_qs(i)
    Z += np.min(Z)
    Z *= 0.05
    ax.plot_surface(X, Y, Z + (n + 1 - i) * 0.9,
                    cmap=cm._colormaps.get_cmap("Spectral_r"))
    label = f"$n={i}$"
    ax.text2D(0.05, 1. - 1/n * i + 0.2, label, transform=ax.transAxes)

ax.set_zticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect("equal")
plt.title("Cosine basis functions for the n-th wavenumber in x- and y-direction")

plt.show()
