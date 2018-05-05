'''
Matrix and Vector
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 0],
              [np.sqrt(1 / 2), np.sqrt(1 / 2)],
              [0, 1],
              [-np.sqrt(1 / 2), np.sqrt(1 / 2)],
              [-1, 0],
              [-np.sqrt(1 / 2), -np.sqrt(1 / 2)],
              [0, -1],
              [np.sqrt(1 / 2), -np.sqrt(1 / 2)]]
             ).T

M = [[1, 2],
     [2, 1]]

y = np.dot(M, x)

print('Determinant = %.2f' % np.linalg.det(M))

print(x)
print(y)

plt.figure()
for index, value in enumerate(x.T):
    if index == 0:
        plt.plot([0, value[0]], [0, value[1]], '-b>')
    elif index == 2:
        plt.plot([0, value[0]], [0, value[1]], '-b^')
    else:
        plt.plot([0, value[0]], [0, value[1]], '-b.')

for index, value in enumerate(y.T):
    if index == 0:
        plt.plot([0, value[0]], [0, value[1]], '-g>')
    elif index == 2:
        plt.plot([0, value[0]], [0, value[1]], '-g^')
    else:
        plt.plot([0, value[0]], [0, value[1]], '-g.')

plt.show()
