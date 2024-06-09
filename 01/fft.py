import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 1000)
f = np.sin(2*np.pi*x) + np.cos(2*np.pi*10*x)
# plt.plot(x, f)
# plt.show()


spectrum = np.fft.fft(f)
print(spectrum)
plt.plot(spectrum)
plt.show()
