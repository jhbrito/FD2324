# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.random import default_rng as default_rng

# simple plot
x = np.arange(0, 5, 0.25)
plt.plot(x, x**2, 'g', label='Square', linewidth=2.5)
plt.title("Function")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-1, 6])
plt.ylim([-10, 150])
plt.legend()
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[0, "min", None, "mid", None, "Max"])
plt.show()
plt.close()

# various lines in the same plot
plt.plot(x, x**2, 'g')
plt.plot(x, x**3, 'r')
plt.plot(x, x**2, 'go')
plt.plot(x, x**3, 'rx')
plt.title("Various functions")
plt.show()
plt.close()

y = np.stack((x**2, x**3)).T
plt.plot(x, y)
plt.title("Various functions")
plt.show()
plt.close()

plt.plot(x, x**2, 'r', x, x**3, 'g')
plt.title("Various functions")
plt.show()
plt.close()

# types of plots
# https://matplotlib.org/stable/plot_types/index.html

plt.scatter(x, x**2)
plt.title("Scatter Plot")
plt.show()
plt.close()

plt.bar(x, x**2)
plt.bar(x, x**3, bottom=x**2)
plt.title("Bar Plot")
plt.show()
plt.close()

plt.stem(x)
plt.title("Stem Plot")
plt.show()
plt.close()

plt.step(x, x**2)
plt.title("Step Plot")
plt.show()
plt.close()

plt.fill_between(x, x**2, x**3)
plt.title("Fill Between Plot")
plt.show()
plt.close()

plt.boxplot((x, x**2))
plt.title("Box Plot")
plt.show()
plt.close()

plt.violinplot((x, x**2))
plt.title("Violin Plot")
plt.show()
plt.close()

plt.polar((x, x**2))
plt.title("Polar Plot")
plt.show()
plt.close()

plt.pie(x)
plt.title("Pie Chart")
plt.show()
plt.close()

plt.hist(x**2)
plt.title("Histogram")
plt.show()
plt.close()

plt.errorbar(x, x**2, (x**2)/10)
plt.title("Error Bar")
plt.show()
plt.close()

# 3D surface with color map
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure(1)
axs = plt.subplot(1, 1, 1, projection='3d')
surf = axs.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.set_zlabel("z")
axs.set_title("3D Surface")
plt.show()
plt.close()


plt.imshow(Z, cmap=cm.coolwarm)
plt.title("imshow")
plt.show()
plt.close()

plt.pcolormesh(X, Y, Z, cmap=cm.coolwarm)
plt.title("pcolormesh")
plt.show()
plt.close()

plt.contour(X, Y, Z, cmap=cm.coolwarm)
plt.title("contour")
plt.show()
plt.close()

plt.contourf(X, Y, Z, cmap=cm.coolwarm)
plt.title("contourf")
plt.show()
plt.close()

X, Y = np.meshgrid([1, 2, 3, 4], [1, 2, 3, 4])
angle = np.pi / 180 * np.array([[15., 30, 35, 45],
                                [25., 40, 55, 60],
                                [35., 50, 65, 75],
                                [45., 60, 75, 90]])
amplitude = np.array([[5, 10, 25, 50],
                      [10, 15, 30, 60],
                      [15, 26, 50, 70],
                      [20, 45, 80, 100]])
U = amplitude * np.sin(angle)
V = amplitude * np.cos(angle)

plt.subplot(1, 3, 1)
plt.barbs(X, Y, U, V)
plt.title("barbs")
plt.subplot(1, 3, 2)
plt.quiver(X, Y, U, V)
plt.title("quiver")
plt.subplot(1, 3, 3)
plt.streamplot(X, Y, U, V)
plt.title("streamplot")
plt.show()
plt.close()


# make data:
rng = default_rng(1)

x = rng.uniform(-3, 3, 256)
y = rng.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
levels = np.linspace(z.min(), z.max(), 7)

plt.subplot(2, 2, 1)
plt.plot(x, y, 'o', markersize=2, color='lightgrey')
plt.tricontour(x, y, z, levels=levels)

plt.subplot(2, 2, 2)
plt.plot(x, y, 'o', markersize=2, color='lightgrey')
plt.tricontourf(x, y, z, levels=levels)

plt.subplot(2, 2, 3)
plt.plot(x, y, 'o', markersize=2, color='lightgrey')
plt.tripcolor(x, y, z)

plt.subplot(2, 2, 4)
plt.plot(x, y, 'o', markersize=2, color='lightgrey')
plt.triplot(x, y)

plt.show()
