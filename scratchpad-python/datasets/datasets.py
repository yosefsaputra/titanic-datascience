import numpy as np


def getErrorSinusoidal(n):
    x = np.random.uniform(0, 1, n)
    y1 = np.sin(x * 2 * np.pi)
    y2 = np.random.normal(0, 0.09, n)
    y = y1 + y2
    return x, y
