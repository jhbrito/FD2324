# Random
import numpy as np
from numpy.random import default_rng as default_rng

rng = default_rng()  # instantiate generator; default BitGenerator is PCG64
a = rng.standard_normal(6)
print("a:{}".format(a))

rng1 = default_rng(seed=1111)
b = rng1.standard_normal(6)
print("b:{}".format(b))

rng2 = default_rng(seed=1111)
c = rng2.standard_normal(6)
print("c:{}".format(c))

del a, b, c, rng, rng1, rng2

rng = default_rng(1)

a = rng.integers(low=0, high=10, size=5, dtype=np.uint8, endpoint=True)
print("a:{}".format(a))

b = rng.random(size=10, dtype=np.float32)  # uniform [0 1[
print("b:{}".format(b))

c = rng.bytes(5)
print("c:{} - {} {} {} {} {}".format(c, c[0], c[1], c[2], c[3], c[4]))

d = ["A", "B", "C", "D", "E"]
p = [0.1, 0.1, 0.5, 0.2, 0.1]
e = rng.choice(d, size=3, replace=False, p=p, axis=0, shuffle=True)  # defaults to uniform distribution
print("e:{}".format(e))

d = ["A", "B", "C", "D", "E"]
print("d:{}".format(d))
rng.shuffle(d)
print("d:{}".format(d))

d = ["A", "B", "C", "D", "E"]
print("d:{}".format(d))
f = rng.permutation(d)
print("f:{}".format(f))

del rng, a, b, c, d, e, f, p

# Distributions - https://numpy.org/doc/stable/reference/random/generator.html
rng = default_rng(1)

a = rng.uniform(low=0, high=10, size=5)
print("a:{}".format(a))

b = rng.standard_normal(size=5)
print("b:{}".format(b))

c = rng.standard_cauchy(size=5)
print("c:{}".format(c))

