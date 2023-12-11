# regression to convert from Celsius to Fahrenheit
# Straightforward example without normalization
# the network should approximate the correct relationship between Celsius and Fahrenheit
# C × 1.8 + 32 = F

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPRegressor
import numpy as np
from numpy.random import default_rng

EPOCHS = 500
lr = 0.1

rng = default_rng(1)

n_samples = 100
samples_mean = 20
samples_std = 5
noise_std = 0
noise_mean = 0

m = 1.8
c = 32
print("True coefficients:{} {}".format(m, c))

celsius = rng.standard_normal(n_samples) * samples_std + samples_mean
fahrenheit = m * celsius + c
noise = rng.standard_normal(n_samples) * noise_std + noise_mean
fahrenheit = fahrenheit + noise

##########################
# Simple Model

model = MLPRegressor(hidden_layer_sizes=(), activation='identity', solver='adam', learning_rate_init=lr)

weights_history = []
bias_history = []
loss_history_simple = []
for i in range(EPOCHS):
    model.partial_fit(celsius.reshape((n_samples, 1)), fahrenheit)
    if i == 0:
        params_begin = [model.coefs_[0][0, 0], model.intercepts_[0][0]]
        print("Simple Model - layer variables init: {}".format(params_begin))
    weights_history.append(model.coefs_[0][0, 0])
    bias_history.append(model.intercepts_[0][0])
    loss_history_simple.append(model.loss_)
print("Finished training the simple model ")
params_end = [model.coefs_[0][0, 0], model.intercepts_[0][0]]
print("Simple Model - Layer variables end: {}".format(params_end))

half_range_weight = 3
weight = np.arange(1.8 - half_range_weight, 1.8 + half_range_weight, half_range_weight / 10.0)
half_range_bias = 18
bias = np.arange(-2, 34, half_range_bias / 10.0)
weight_grid_3D, bias_grid_3D, celsius_grid_3D = np.meshgrid(weight, bias, celsius)
squared_error = ((celsius_grid_3D * weight_grid_3D + bias_grid_3D) - (celsius_grid_3D * 1.8 + 32)) ** 2
mean_squared_error = np.mean(squared_error, axis=2)
weight_grid_2D, bias_grid_2D = np.meshgrid(weight, bias)

fig = plt.figure(1)
ax = fig.add_subplot(1, 2, 1, projection='3d')
# surf = ax.plot_surface(weight_grid_2D, bias_grid_2D, mean_squared_error, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0.0, 20000.0)
contour = ax.contour3D(weight_grid_2D, bias_grid_2D, mean_squared_error, 25, cmap=cm.coolwarm, antialiased=True)
# fig.colorbar(contour, shrink=0.5, aspect=5)
line = ax.plot(weights_history, bias_history, loss_history_simple, 'g-', linewidth=2, antialiased=False)
scatter = ax.scatter([1.8], [32], [0], c='r', marker='.')
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Loss")
ax.set_title("Simple Model")

c = np.array([20.0])
f = model.predict(c.reshape((1, 1)))
print("Simple model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = c * 1.8 + 32
print("Simple model error is: {} degrees Fahrenheit".format(f - f_gt))

######################
# Complex Model

model = MLPRegressor(hidden_layer_sizes=(4, 4), activation='identity', solver='adam', learning_rate_init=lr)
loss_history_complex = []
for i in range(EPOCHS):
    model.partial_fit(celsius.reshape((n_samples, 1)), fahrenheit)
    loss_history_complex.append(model.loss_)
print("Finished training the complex model")

c = np.array([20.0])
f = model.predict(c.reshape((1, 1)))
print("Complex model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = c * 1.8 + 32
print("Complex model error is: {} degrees Fahrenheit".format(f - f_gt))

print("Complex layer variables")
print(" l0 weights and bias: {} {}".format(model.coefs_[0], model.intercepts_[0]))
print(" l1 weights and bias: {} {}".format(model.coefs_[1], model.intercepts_[1]))
print(" l2 weights and bias: {} {}".format(model.coefs_[2], model.intercepts_[2]))


##########################################
# Normalization

def normalize(values):
    values_std = np.std(values)
    values_mean = np.mean(values)
    values_n = (values - values_mean) / values_std
    return (values_n, values_mean, values_std)


def denormalize(values_n, values_mean, values_std):
    values_u = values_n * values_std + values_mean
    return values_u


celsius_n, celsius_mean, celsius_std = normalize(celsius)
fahrenheit_n, fahrenheit_mean, fahrenheit_std = normalize(fahrenheit)

model = MLPRegressor(hidden_layer_sizes=(), activation='identity', solver='adam', learning_rate_init=lr)

weights_history = []
bias_history = []
loss_history_normalized = []
for i in range(EPOCHS):
    model.partial_fit(celsius_n.reshape((n_samples, 1)), fahrenheit_n)
    if i == 0:
        params_begin = [model.coefs_[0][0, 0], model.intercepts_[0][0]]
        print("Normalized Model - layer variables init: {}".format(params_begin))
    weights_history.append(model.coefs_[0][0, 0])
    bias_history.append(model.intercepts_[0][0])
    loss_history_normalized.append(model.loss_)
print("Finished training the normalized model ")
params_end = [model.coefs_[0][0, 0], model.intercepts_[0][0]]
print("Normalized Model - Layer variables end: {}".format(params_end))

weight_n = np.arange(1 - 0.5, 1 + 0.5, 0.01)
bias_n = np.arange(0 - 0.5, 0 + 0.5, 0.01)
weight_grid_3D_n, bias_grid_3D_n, celsius_grid_3D_n = np.meshgrid(weight_n, bias_n, celsius_n)
squared_error_n = ((celsius_grid_3D_n * weight_grid_3D_n + bias_grid_3D_n) - ((denormalize(celsius_grid_3D_n, celsius_mean, celsius_std) * 1.8 + 32 - fahrenheit_mean) / fahrenheit_std)) ** 2
mean_squared_error_n = np.mean(squared_error_n, axis=2)
weight_grid_2D_n, bias_grid_2D_n = np.meshgrid(weight_n, bias_n)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlim(0.5, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.5)

# surf = ax.plot_surface(weight_grid_2D_n, bias_grid_2D_n, mean_squared_error_n, cmap=cm.coolwarm, linewidth=0, antialiased=True)
contour = ax.contour3D(weight_grid_2D_n, bias_grid_2D_n, mean_squared_error_n, 25, cmap=cm.coolwarm, antialiased=True)
# fig.colorbar(contour, shrink=0.5, aspect=5)
line = ax.plot(weights_history, bias_history, loss_history_normalized, 'g-', linewidth=2)
# line = ax.scatter(weight_history_n, bias_history_n, loss_history_n, cmap=cm.coolwarm, linewidth=1)
scatter = ax.scatter([1], [0], [0], c='r', marker='.')
ax.set_xlabel("Normalized Weight")
ax.set_ylabel("Normalized Bias")
ax.set_zlabel("Normalized Loss")
ax.set_title("Normalized Model")

plt.show()

c = np.array([20.0])
f_gt = c * 1.8 + 32
c = (c - celsius_mean) / celsius_std

c = c.reshape((1, 1))

f = model.predict(c)
f = denormalize(f, fahrenheit_mean, fahrenheit_std)
print("Normalized model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
print("Normalized model error is: {} degrees Fahrenheit".format(f - f_gt))

#############################
# Loss vs Epoch

plt.figure(3)
plt.subplot(1, 3, 1)
plt.plot(loss_history_simple)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Simple Model")

plt.subplot(1, 3, 2)
plt.plot(loss_history_complex)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Complex Model")

plt.subplot(1, 3, 3)
plt.plot(loss_history_normalized)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Normalized Model")

plt.show()

plt.figure(4)
plt.subplot(1, 3, 1)
plt.plot(loss_history_simple)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Simple Model")
plt.xlim([0, 500])
plt.ylim([0, 100])

plt.subplot(1, 3, 2)
plt.plot(loss_history_complex)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Complex Model")
plt.xlim([0, 500])
plt.ylim([0, 100])

plt.subplot(1, 3, 3)
plt.plot(loss_history_normalized)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Normalized Model")
plt.xlim([0, 50])
plt.ylim([0, 1])

plt.show()
