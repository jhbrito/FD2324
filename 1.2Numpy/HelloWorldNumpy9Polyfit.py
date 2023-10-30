# Polynomial fit
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng(1)

x = np.array(range(-10, 10)) * 0.1
a = [2, 3, 4, 5]
y = a[0] * x ** 3 + a[1] * x ** 2 + a[2] * x + a[3]

a_est = np.polyfit(x, y, 3)
print(a_est)
y_est = a_est[0] * x ** 3 + a_est[1] * x ** 2 + a_est[2] * x + a_est[3]

plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_est, 'g', label='Fitted polynomial')
plt.legend()
plt.show()

# Now with noise
noise_std = 0.5
noise = rng.standard_normal(len(y)) * noise_std
y_with_noise = y + noise

a_est = np.polyfit(x, y_with_noise, 3)
print(a_est)

y_est = a_est[0] * x ** 3 + a_est[1] * x ** 2 + a_est[2] * x + a_est[3]

plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_with_noise, 'rx', label='Noisy data', markersize=10)
plt.plot(x, y_est, 'g', label='Fitted polynomial')
plt.legend()
plt.show()
