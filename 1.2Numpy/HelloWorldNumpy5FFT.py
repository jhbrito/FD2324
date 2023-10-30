import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal as scipy_signal

N = 1000  # N total samples
fs = 1000  # sampling frequency = 1kHz
Ts = 1/fs  # Sampling period = 1 ms
f_base = fs / N
# T_total = N * Ts
# f_base = 1 / T_total

t = np.arange(N, dtype=np.float32) * Ts  # time vector
f = np.arange(N, dtype=np.float32) * f_base  # frequency  vector

A1 = 1
f1 = 10.0  # signal 1 frequency = 10Hz

A2 = 0.1
f2 = 100.0  # signal 2 frequency = 100Hz

signal1 = A1 * np.sin(2 * np.pi * f1 * t)
signal2 = A2 * np.sin(2 * np.pi * f2 * t)

signal = signal1 + signal2
plt.plot(t, signal)
plt.title("Signal")
plt.xlabel("t(s)")
plt.show()

spectrum = np.fft.fft(signal)
# freq = np.fft.fftfreq(t.shape[-1])
plt.plot(f, np.abs(spectrum))
plt.title("Spectrum Magnitude")
plt.xlabel("f(Hz)")
plt.show()

plt.plot(f[0:int(len(f)/2)], np.abs(spectrum)[0:int(len(f)/2)])
plt.title("Spectrum Magnitude half")
plt.xlabel("f(Hz)")
plt.show()

freq = np.fft.fftfreq(t.shape[-1], d=Ts)  # frequency  vector for plotting
plt.plot(freq, np.abs(spectrum))
plt.title("Spectrum Magnitude")
plt.xlabel("f(Hz)")
plt.show()

signal_reconstructed = np.fft.ifft(spectrum)
plt.plot(t, signal_reconstructed)
plt.title("Reconstructed Signal")
plt.xlabel("t(s)")
plt.show()

# For higher dimensions: fft2, ifft2, fftn, ifftn

# Convolution - filter is flipped
signal = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
filter = np.array([0.0, 1.0, 2.0])
print("Full filtered_signal:{}".format(np.convolve(signal, filter)))
print("Same filtered_signal:{}".format(np.convolve(signal, filter, mode='same')))
print("Valid filtered_signal:{}".format(np.convolve(signal, filter, mode='valid')))

print("fftconvolve:{}".format(scipy_signal.fftconvolve(signal, filter)))

signal = np.random.standard_normal(10000000)
filter = np.random.standard_normal(10000)

start = time.time()
b = scipy_signal.fftconvolve(signal, filter)
end = time.time()
print("Elapsed time (fftconvolve) = %s seconds" % (end - start))

start = time.time()
a = np.convolve(signal, filter)
end = time.time()
print("Elapsed time (convolve) = %s seconds" % (end - start))
