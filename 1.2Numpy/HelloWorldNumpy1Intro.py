# Numpy is a subset of SciPy
# https://numpy.org/doc/stable/reference/
# https://numpy.org/doc/stable/reference/routines.html
import numpy as np

# Simple Vector
a = np.array([1, 2, 3])  # array from list
print("a:{}".format(a))

a = np.array((1, 2, 3))  # array from tuple
print("a:{}".format(a))

# type and indexing
print("type:{}".format(type(a)))
print("a[0]:{}".format(a[0]))

# Simple Matrix and indexing
a = np.array([[1, 2, 3], [4, 5, 6]])
print("a:{}".format(a))
print("a[0,1]:{}".format(a[0, 1]))
print("a[0][1]:{}".format(a[0][1]))

print("a:{}".format(a))
a = np.arange(15).reshape(3, 5)

# arrays with types
a = np.ndarray((10, 10), dtype=np.uint8)
print("a:{}".format(a))
print("type:{}".format(type(a)))
print("a[0][0]:{}".format(a[0, 0]))

# Special matrices creation
a = np.empty((3, 4))
print("a:{}".format(a))
b = np.zeros((3, 4))
print("b:{}".format(b))
c = np.ones((3, 4))
print("c:{}".format(c))
d = np.eye(3)
print("d:{}".format(d))
e = np.random.randn(12).reshape((3, 4))
print("e:{}".format(e))

# data types: https://numpy.org/doc/stable/user/basics.types.html
# bool_, byte, ubyte, short, ushort, intc, uintc, int_, uint, longlong, ulonglong,
# half / float16, single, double, longdouble,
# csingle, cdouble, clongdouble
#
# large list of aliases: https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases
a = np.ushort(10)
print("a:{}".format(a))
ai = np.iinfo(np.ushort)
print("ushort bits:", ai.bits)
print("ushort min:", ai.min)
print("ushort max:", ai.max)

a = np.float16(1)
b = a + np.finfo(np.float16).eps
print("b:{}".format(b))

# Constants (IEEE 754)
a = np.nan
if np.isnan(a):
    print("a is a NaN ({})".format(a))
b = np.inf
if b > 9999:
    print("b is greater than 9999 ({})".format(b))
c = -np.inf
if c < -9999:
    print("c is lower than -9999 ({})".format(c))

print("Euler’s constant:{}".format(np.e))
print("Euler-Mascheroni constant:{}".format(np.euler_gamma))
print("Pi:{}".format(np.pi))

# Matrix properties
a = np.empty((3, 4))
print("a.shape:{}".format(a.shape))
print("a.ndim:{}".format(a.ndim))
print("a.dtype:{}".format(a.dtype))
print("a.dtype.name:{}".format(a.dtype.name))
print("a.itemsize:{}".format(a.itemsize))
print("a.size:{}".format(a.size))
print("a:{}".format(a))
