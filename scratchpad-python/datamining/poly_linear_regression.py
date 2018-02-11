'''
This scratch pad is to visualize Linear Regression models with increasing number of polynomial degree.
With low number of degree, the model is not good enough to model the data. As it increases, it tends to reduce the training RMSE and test RMSE.
However, when the overfitting happens, the training RMSE reaches its lowest but the test RMSE shoots up high.
'''


import matplotlib.pyplot as plt
import numpy as np
import datasets.datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

degree_list = [i if i > 2 else i for i in range(1, 13)]
subplot_row = 3
subplot_col = 4

x, y = datasets.datasets.getErrorSinusoidal(20)
x = x[:, None]

x_test, y_test = datasets.datasets.getErrorSinusoidal(5)
x_test = x_test[:, None]

rmse_train_list = []
rmse_test_list = []

f, axarr = plt.subplots(subplot_row, subplot_col)
f.suptitle('Polynomial Linear Regression')
for seq, degree in enumerate(degree_list):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    x_test_poly = poly.fit_transform(x_test)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_predicted = model.predict(x_poly)
    y_test_predicted = model.predict(x_test_poly)

    x_star = np.linspace(0, 1, 10000)[:, None]
    x_star_poly = poly.fit_transform(x_star)
    y_star = model.predict(x_star_poly)

    # Calculating Erms
    MSE = mean_squared_error(y[:, None], y_predicted)
    RMSE = np.sqrt(MSE)
    rmse_train_list.append(RMSE)

    MSE = mean_squared_error(y_test[:, None], y_test_predicted)
    RMSE = np.sqrt(MSE)
    rmse_test_list.append(RMSE)

    # Plotting
    ax = axarr[int(seq / subplot_col), seq % subplot_col]
    ax.text(0.9,
            0.9,
            s='degree %s' % degree,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.set_ylim(np.min(y) * 1.5, np.max(y) * 1.5)
    ax.scatter(x_star, y_star, s=0.25, label='model')
    ax.scatter(x, y, s=5, label='training')
    ax.scatter(x_test, y_test_predicted, s=5, label='test')
    if seq == 0:
        axarr[int(seq / subplot_col), seq % subplot_col].legend()

    print('Model Coef (M=%s): %s, %s' % (degree, model.intercept_, model.coef_))

plt.figure()
plt.title('RMSE vs degree')
plt.scatter(degree_list, rmse_train_list)
plt.plot(degree_list, rmse_train_list)
plt.scatter(degree_list, rmse_test_list)
plt.plot(degree_list, rmse_test_list)
plt.xlabel('complexity')
plt.ylabel('unit')
plt.show()
