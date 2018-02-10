import matplotlib.pyplot as plt
import numpy as np


def integrate(func, x, delta, value_prev=0):
    return value_prev + delta * (func(x))


def func(x):
    return x * x


x_range = [i / 1000 for i in range(0, 10000)]
delta = x_range[1] - x_range[0]
y_range = [func(i) for i in x_range]
y2_range = []

for sequence, x in enumerate(x_range):
    if sequence == 0:
        value_prev = 0
    else:
        value_prev = y2_range[sequence-1]

    y2_range.append(integrate(func, x, delta, value_prev))

plt.plot(x_range, y_range, label='func(x)')
plt.plot(x_range, y2_range, label='F(x)')
plt.legend()
plt.show()
