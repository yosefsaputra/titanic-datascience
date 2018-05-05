'''
Plotting of Vectors
'''


import matplotlib.pyplot as plt
import numpy as np

x = np.array([2, 7])
y = np.array([5, 3])

plt.figure()
for index, value in enumerate(x):
    plt.plot([0, x[index]], [0, y[index]], '-bo')

M = np.array([[0.5, 0.1],
              [0.2, 0.5]])

input = np.array([x, y])

result = np.dot(M, input)

print(result[0, :])
for index, value in enumerate(result[0, :]):
    print(index)
    plt.plot([0, result[0, :][index]], [0, result[1, :][index]], '-go')

plt.xlim([0, 120])
plt.ylim([0, 120])
plt.show()
