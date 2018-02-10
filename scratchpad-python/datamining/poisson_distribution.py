import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def poissonDist(x, lambda_poisson):
    return np.power(lambda_poisson, x) * np.exp(-lambda_poisson) / scipy.special.factorial(x)

lambda_poisson_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,20,30,40,50,60,70,80,90,100]

fig = plt.figure()
for lambda_poisson in lambda_poisson_list:
    x = np.arange(0, 150, 0.01)
    y = poissonDist(x, lambda_poisson)
    plt.scatter(x, y)

plt.show()
