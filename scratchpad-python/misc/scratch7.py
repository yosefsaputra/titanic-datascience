import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def submission(x, y):
    z = 1 - x - y
    return z


def multiplication(x, y):
    z = 0.024 / x / y
    return z


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, 1.0, 0.05)
X, Y = np.meshgrid(x, y)

zs_mul = np.array([multiplication(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
zs_sub = np.array([submission(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])

zs = zs_mul
for seq, value in enumerate(zs):
    if value < zs_sub[seq]:
        zs[seq] = None
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)

zs = zs_sub
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)

zs = zs_mul
for seq, value in enumerate(zs):
    if value > zs_sub[seq]:
        zs[seq] = None
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.show()
