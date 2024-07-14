# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from matplotlib import cm
from mpi4py import MPI
plt.rcParams['text.usetex'] = True
spectral = cm._colormaps.get_cmap("Spectral_r")

nb_grid_pts = (512*8, 512*8, 2)
fft = FFT(nb_grid_pts, engine='pfft', communicator=MPI.COMM_WORLD)
dim = 3


def curl_fourier(u_cxyz):
    """Computes the curl of a vector field in real space."""
    u_hat_c_xyz = fft.fourier_space_field('u_c hat', dim)
    fft.fft(u_cxyz, u_hat_c_xyz)
    (k_x, k_y, k_z) = (fft.fftfreq[0], fft.fftfreq[1], fft.fftfreq[2])
    (u_x, u_y, u_z) = (u_hat_c_xyz.p[0], u_hat_c_xyz.p[1], u_hat_c_xyz.p[2])
    dyuz = 1j * k_y * u_z
    dzuy = 1j * k_z * u_y
    dzux = 1j * k_z * u_x
    dxuz = 1j * k_x * u_z
    dxuy = 1j * k_x * u_y
    dyux = 1j * k_y * u_x

    res = fft.fourier_space_field('res', dim)
    resr = fft.real_space_field('resr', dim)
    res.p[0] = dyuz - dzuy
    res.p[1] = dzux - dxuz
    res.p[2] = dxuy - dyux
    fft.ifft(res, resr)
    return resr.p


# assert that curl vanishes for constant field
u_cxyz = np.ones([3, *fft.nb_subdomain_grid_pts])
curlu_cxyz = curl_fourier(u_cxyz)
np.testing.assert_allclose(curlu_cxyz, 0)


# plot curl for constant rotation
norm = np.array([0, 0, 1])
translated_coords = (fft.coords - 0.5)
u_cxyz = np.cross(norm, translated_coords, axis=0)
np.testing.assert_allclose(u_cxyz[2, :, :, 0], 0)

curlu_cxyz = curl_fourier(u_cxyz)
# assert that curl only has z component
np.testing.assert_allclose(curlu_cxyz[0, :, :, 0], 0)
np.testing.assert_allclose(curlu_cxyz[1, :, :, 0], 0)
# assert that z-components are the same along z-axis
np.testing.assert_allclose(curlu_cxyz[1, :, :, 0], curlu_cxyz[1, :, :, -1])

# plot the z component
arrow_subsampling = int(nb_grid_pts[0]/16)
X, Y = np.meshgrid(
    translated_coords[0, :, 0, 0], translated_coords[1, 0, :, 0])
X_s, Y_s = np.meshgrid(
    translated_coords[0, 0:-1:arrow_subsampling, 0, 0], translated_coords[1, 0, 0:-1:arrow_subsampling, 0])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.quiver(X_s, Y_s, u_cxyz[1, 0:-1:arrow_subsampling, 0:-1:arrow_subsampling,
           0], u_cxyz[0, 0:-1:arrow_subsampling, 0:-1:arrow_subsampling, 0])
cont = ax2.pcolormesh(
    X, Y, curlu_cxyz[2, :, :, 0], cmap=spectral, shading="nearest")

# describe the plot and make it pretty
fig.suptitle("Curl of a rigid body rotation computed on a " +
             str(nb_grid_pts[0])+r"$\times$"+str(nb_grid_pts[1])+r" grid in the Fourier domain")
ax1.set_title(
    r"$\vec{v}_{xy}$ for a rotation with $\omega=(0,0,2)^T$ around the origin (subsampled$\times$"+str(arrow_subsampling)+r")")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_title(
    r"$z$-component of $\nabla\times\vec{v}_{xy}$ ($x$ and $y$ components are zero)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax1.set_aspect("equal")
ax2.set_aspect("equal")
cbar = fig.colorbar(cont, orientation="horizontal", pad=0.1)
cbar.ax.set_xlabel(r"$||\nabla\times\vec{v}_{xy}||$")
# plt.show()
plt.savefig("curl.png", dpi=400)
