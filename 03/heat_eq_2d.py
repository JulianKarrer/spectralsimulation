# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from matplotlib import cm
from mpi4py import MPI
plt.rcParams['text.usetex'] = True

nb_grid_pts = (4000, 2000)
length = (20., 10.)
nx, ny = nb_grid_pts
lx, ly = length
fft = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)
avg_temp = 300   # average temperature
source_sink_temp = 5
source_sink_radius = 0.6

qs = fft.real_space_field('qs')
x, y = fft.coords
qs_src = (lx/4, ly/2)
qs_snk = (3*lx/4, ly/2)
qs.p[np.sqrt((x*lx-qs_src[0])**2 + (y*ly-qs_src[1])**2)
     <= source_sink_radius] = source_sink_temp
qs.p[np.sqrt((x*lx-qs_snk[0])**2 + (y*ly-qs_snk[1])**2)
     <= source_sink_radius] = -source_sink_temp

qs_hat = fft.fourier_space_field('qs hat')
fft.fft(qs, qs_hat)

k_x = fft.fftfreq[0] * 2*np.pi * (nx/lx)
k_y = fft.fftfreq[1] * 2*np.pi * (ny/ly)
k_2 = k_x*k_x + k_y*k_y
# k_2[0, 0] = 1  # avoid division by zero


res = fft.real_space_field('res')
res_hat = fft.fourier_space_field('res hat')
res_hat = qs_hat.p / (k_2) * fft.normalisation
res_hat[0, 0] = avg_temp

fft.ifft(res_hat, res)

# plot the solution
x = np.arange(nx)*lx/nx
y = np.arange(ny)*ly/ny
X, Y = np.meshgrid(x, y)
Z = res.p.T

# plt.plot(x, Z[int(ny/2), :])
# plt.show()


min, max = np.min(Z), np.max(Z)
print(min, max)
cont = plt.pcolormesh(X, Y, Z, cmap=cm._colormaps.get_cmap(
    "Spectral_r"), shading="nearest")
plt.gcf().set_size_inches(10, 5)
plt.gca().set_aspect(1)
plt.xlim(0, lx), plt.ylim(0, ly)
plt.suptitle(r"Stationary distribution of Temperature")
plt.title(
    r"$T_{avg}="+str(avg_temp)+r"K$, $q_s ="+str(source_sink_temp)+r" \Theta \left("+str(source_sink_radius)+r"-\left\vert["+str(qs_src[0])+r", "+str(qs_src[1])+r"]^T - [x,y]^T]\right\vert^2\right) - "+str(source_sink_temp)+r" \Theta \left("+str(source_sink_radius)+r"-\left\vert["+str(qs_snk[0])+r", "+str(qs_snk[1])+r"]^T - [x,y]^T]\right\vert^2\right)$")
plt.figtext(
    0.5, 0.02, r"periodic boundary conditions on domain $[0;"+str(int(lx))+r"]\times [0;"+str(int(ly))+r"$] using $"+str(nx)+r"\times "+str(ny)+r"$ grid points", ha="center", fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(cont)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Temperature (K)', rotation=90)
plt.savefig("heat_equation_2d.png")
