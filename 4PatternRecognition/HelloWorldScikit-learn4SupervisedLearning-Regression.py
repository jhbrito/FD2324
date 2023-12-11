import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

from sklearn import metrics
from numpy.random import default_rng

rng = default_rng(1)

n_samples = 100
mean_samples = 20.0
std_samples = 5.0

celsius = rng.standard_normal(n_samples)*std_samples+mean_samples
fahrenheit = celsius * 1.8 + 32
celsius = celsius.reshape((len(celsius), 1))

noise_std = 1.0
noise = rng.standard_normal(len(fahrenheit)) * noise_std

fahrenheit = fahrenheit + noise


fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 1)
ax.scatter(celsius[:, 0], fahrenheit, c='r', marker='x')
ax.set_xlabel("Celsius")
ax.set_ylabel("Fahrenheit")
ax.set_title("Data")

X_train, X_test, Y_train, Y_test = train_test_split(celsius, fahrenheit, test_size=0.2, random_state=1)

ax = fig.add_subplot(1, 3, 2)

ax.scatter(X_train[:, 0], Y_train, c='r', marker='x')
ax.scatter(X_test[:, 0], Y_test, c='m', marker='x')

ax.set_xlabel("Celsius")
ax.set_ylabel("Fahrenheit")
ax.set_title("Split Data")

model = LinearSVR()
# model = NuSVR()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = AdaBoostRegressor()
# model = MLPRegressor()
# model = SGDRegressor()

model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

ax = fig.add_subplot(1, 3, 3)
ax.scatter(X_test[:, 0], Y_predict, c='r', marker='*')
ax.set_xlabel("Celsius")
ax.set_ylabel("Fahrenheit")
ax.set_title("Predicted Values")

plt.show()
plt.close(fig)

mse = metrics.mean_squared_error(Y_test, Y_predict)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(Y_test, Y_predict)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(Y_test, Y_predict)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(Y_test, Y_predict)
print("Median Absolute Error:", mdae)
