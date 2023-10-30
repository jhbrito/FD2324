# Math functions - https://numpy.org/doc/stable/reference/routines.math.html
import numpy as np

a = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
b = np.radians(a)

print("a:{}".format(a))
print("b:{}".format(b))

print("sin:{}".format(np.sin(b)))
print("cos:{}".format(np.cos(b)))

a = np.array([0.1, 0.2, 0.5, 0.6, 0.9])
print("round_:{}".format(np.round_(a)))
print("floor:{}".format(np.floor(a)))
print("ceil:{}".format(np.ceil(a)))
print("sum:{}".format(np.sum(a)))
print("prod:{}".format(np.prod(a)))
print("exp:{}".format(np.exp(a)))
print("log:{}".format(np.log(a)))
print("sqrt:{}".format(np.sqrt(a)))
