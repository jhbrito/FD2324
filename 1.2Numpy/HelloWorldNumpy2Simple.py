import numpy as np
import os

folder = "../Files"


# Broadcasting
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 2.0, 2.0, 2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("a*b:{}".format(a*b))
print("a**b:{}".format(a**b))

b = np.array([2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("a*b:{}".format(a*b))
print("a**b:{}".format(a**b))

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, 2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("b+a:{}".format(b+a))
print("a*b:{}".format(a*b))
print("b*a:{}".format(b*a))
print("a**b:{}".format(a**b))

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[1.0], [2.0]])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("b+a:{}".format(b+a))
print("a*b:{}".format(a*b))
print("b*a:{}".format(b*a))
print("a**b:{}".format(a**b))

del a, b

# Statistics
a = np.array([1.0, 2.0, 3.0, 4.0, 3.0])
print("Max:{}".format(np.max(a)))
print("Min:{}".format(np.min(a)))
print("Mean:{}".format(np.mean(a)))
print("NanMean:{}".format(np.nanmean(a)))  # ignores NaNs
print("Std Dev:{}".format(np.std(a)))
print("Median:{}".format(np.median(a)))
print("Variance:{}".format(np.var(a)))
print("Cross-correlation:{}".format(np.correlate([1, 2, 3], [0, 1, 0.5])))

data = [0, 1, 2, 1, 0, 0]
h = np.histogram(data, bins=[0, 1, 2, 3])
print("Histogram with bins", h)

h = np.histogram(data, bins='auto')
print("Histogram with auto bins", h)

del a, data, h  # cleanup

# files
a = np.array([[1, 2, 3], [4, 5, 6]])
np.save(os.path.join(folder, "a"), a)
del a
a = np.load(os.path.join(folder, "a.npy"))
print("a:", a)

b = np.array([[1, -2], [0, 1], [1, 0]])
np.savez(os.path.join(folder, "ab"), a, b)
del a, b
npzfile = np.load(os.path.join(folder, "ab.npz"))
print(npzfile.files)
a = npzfile[npzfile.files[0]]  # or npzfile['arr_0']
b = npzfile[npzfile.files[1]]  # or npzfile['arr_1']
print("a:", a)
print("b:", b)

del a, b, npzfile

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, -2], [0, 1], [1, 0]])
np.savez(os.path.join(folder, "ab"), a=a, b=b)
del a, b
npzfile = np.load(os.path.join(folder, "ab.npz"))
print(npzfile.files)
a=npzfile['a']
b=npzfile['b']
print("a:", a)
print("b:", b)
del npzfile

np.savetxt(os.path.join(folder, "a.txt"), a, delimiter=',')
np.savetxt(os.path.join(folder, "b.txt"), b, fmt='%1.4e')
del a, b
a = np.loadtxt(os.path.join(folder, "a.txt"), delimiter=',')
b = np.loadtxt(os.path.join(folder, "b.txt"))

# cleanup files
os.remove(os.path.join(folder, "a.npy"))
os.remove(os.path.join(folder, "ab.npz"))
os.remove(os.path.join(folder, "a.txt"))
os.remove(os.path.join(folder, "b.txt"))

# bitwise binary
a = np.array([1, 2, 3, -3], dtype=np.int16)
print("a[0]", np.binary_repr(a[0]))
print("a[0]", np.binary_repr(a[0], width=8))
print("a[1]", np.binary_repr(a[1], width=8))
print("a[2]", np.binary_repr(a[2], width=8))
print("a[3]", np.binary_repr(a[3]))
print("a[3] C2", np.binary_repr(a[3], width=8))


def print_binary(name, x, width=8):
    print("{}: {} {} {} {}".format(name, np.binary_repr(x[0], width=width), np.binary_repr(x[1], width=width), np.binary_repr(x[2], width=width), np.binary_repr(x[3], width=width)))


print_binary("a", a)
b = np.array([1, 3, 5, -2], dtype=np.int16)
print_binary("b", b)

c = np.bitwise_and(a, b)
print_binary("c", c)
c1 = a & b
print_binary("c1", c1)

d = np.bitwise_or(a, b)
print_binary("d", d)
d1 = a | b
print_binary("d1", d1)

e = np.bitwise_xor(a, b)
print_binary("e", e)
e1 = a ^ b
print_binary("e1", e1)

f = np.invert(a)
print_binary("f", f)
f1 = ~a
print_binary("f1", f1)

a = np.array([4, 4, 4, 4], dtype=np.int16)
b = np.array([1, 2, -1, -2], dtype=np.int16)
print_binary("a", a)
print_binary("b", b)
g = np.left_shift(a, b)
print_binary("g", g, np.iinfo(np.int16).bits)
g1 = a << b
print_binary("g1", g1, np.iinfo(np.int16).bits)

a = np.array([-5, -5, -5, -5], dtype=np.int16)
print_binary("a", a)
g = np.left_shift(a, b)
print_binary("g", g, np.iinfo(np.int16).bits)
g1 = a << b
print_binary("g1", g1, np.iinfo(np.int16).bits)

h = np.right_shift(a, b)
print_binary("h", h, np.iinfo(np.int16).bits)
h1 = a >> b
print_binary("h1", h1, np.iinfo(np.int16).bits)
