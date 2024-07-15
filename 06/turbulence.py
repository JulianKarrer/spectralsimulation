# type: ignore
import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt
from mpi4py import MPI
from '../05/vortex.py' import *

# mpirun -np 4 python3 vortex.py
# 32x32x32 is fine
n = 32
nb_grid_pts = (n, n, n)
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
nu = 1/1600
dt = 0.001
t_end = 30


wavevector = (2 * np.pi * fft.fftfreq.T / grid_spacing).T
zero_wavevector = (wavevector.T == np.zeros(3, dtype=int)).T.all(axis=0)
wavevector_sq = np.sum(wavevector ** 2, axis=0)
# Fourier space velocity field
random_field = np.zeros((3,) + fft.nb_fourier_grid_pts, dtype=complex)
rng = np.random.default_rng()
random_field.real = rng.standard_normal(random_field.shape)
random_field.imag = rng.standard_normal(random_field.shape)
# Initial velocity field should decay as k^(-5/3) for the Kolmogorov spectrum
fac = np.zeros_like(wavevector_sq)
# Avoid division by zero
fac[np.logical_not(zero_wavevector)] = velocity_amplitude * \
	wavevector_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)
random_field *= fac

