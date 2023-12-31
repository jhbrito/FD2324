# Non linear optimization
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as scipy_optimize
from numpy.random import default_rng

rng = default_rng(1)

x = np.array(range(100)) * 0.1
a = np.array([1.0, 2.0])

a0 = np.array([2, 4])
# a0 = np.array([4000, -2000])

y = a[0] * x + a[1] * x * np.sin(2 * x)

y_initial = a0[0] * x + a0[1] * x * np.sin(2 * x)
plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_initial, 'r', label='Initial Model')
plt.title('Starting point')
plt.legend()
plt.show()


def fun(params):
    return params[0] * x + params[1] * x * np.sin(2 * x) - y


solution = scipy_optimize.least_squares(fun, a0)
a_est = solution.x
print("Estimated parameters", a_est)
print("Initial loss: {}".format(0.5*(np.sum(np.square(fun(a0))))))
print("Final loss: {}".format(0.5*(np.sum(np.square(fun(a_est))))))
y_est = a_est[0] * x + a_est[1] * x * np.sin(2 * x)

plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_est, 'g', label='Fitted function')
plt.legend()
plt.title('nonlinear least-squares')
plt.show()

# Now with noise
noise_std = 0.5
noise = rng.standard_normal(len(y)) * noise_std
y_with_noise = y + noise


def fun_noise(params):
    return params[0] * x + params[1] * x * np.sin(2 * x) - y_with_noise


solution = scipy_optimize.least_squares(fun_noise, a0)  # optionally with bounds
a_est = solution.x
print("With noisy data")
print("Estimated parameters", a_est)
print("Initial loss: {}".format(0.5*(np.sum(np.square(fun_noise(a0))))))
print("Final loss: {}".format(0.5*(np.sum(np.square(fun_noise(a_est))))))

y_est = a_est[0] * x + a_est[1] * x * np.sin(2 * x)

plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_with_noise, 'r', label='Noisy data')
plt.plot(x, y_est, 'g', label='Fitted function')
plt.legend()
plt.title('nonlinear least-squares with noise')
plt.show()
