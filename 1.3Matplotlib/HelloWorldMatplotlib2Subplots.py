import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.25)

# with figures
fig = plt.figure(1)
plt.show()
plt.close(fig)

fig, ax = plt.subplots()  # a figure with a single Axes
plt.show()
plt.close(fig)

fig, axs = plt.subplots(2, 2)
plt.show()
plt.close(fig)

fig, axs = plt.subplots(1, 2)
axs[0].set_title("A")
axs[1].set_title("B")
plt.show()
plt.close(fig)

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("A")
axs[0, 1].set_title("B")
axs[1, 0].set_title("C")
axs[1, 1].set_title("D")
plt.show()
plt.close(fig)

# simple subplot
plt.subplot(1, 2, 1)
plt.plot(x, x**2)
plt.title("Square")

plt.subplot(1, 2, 2)
plt.plot(x, x**3)
plt.xlabel("x")
plt.ylabel("x^3")
plt.title("Cubic")
plt.show()
plt.close()

# with axes
fig = plt.figure(1)
ax = fig.add_subplot(1, 2, 1)
ax.plot(x, x**2)
ax.set_xlabel("x")
ax.set_ylabel("x^2")
ax.set_title("Square")

ax = fig.add_subplot(1, 2, 2)
ax.plot(x, x**3)
ax.set_xlabel("x")
ax.set_ylabel("x^3")
ax.set_title("Cubic")
plt.show()
plt.close(fig)
