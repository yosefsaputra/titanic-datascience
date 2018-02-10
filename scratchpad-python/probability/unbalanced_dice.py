import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import random


def rollUnbalanceDice1():
    x = random.randint(1, 100)
    dice = 1
    if 1 <= x <= 10:
        dice = 1
    elif 11 <= x <= 50:
        dice = 2
    elif 51 <= x <= 60:
        dice = 3
    elif 61 <= x <= 65:
        dice = 4
    elif 66 <= x <= 80:
        dice = 5
    elif 81 <= x <= 100:
        dice = 6
    return dice


def getSamples(n):
    mean = 0
    for i in range(n):
        mean += rollUnbalanceDice1()
    return mean / n


plt.figure()
for j in range(1, 1000):
    x = [getSamples(j) for i in range(100)]

    plt.hist(x, bins=200)
    plt.draw()
    plt.pause(.001)
