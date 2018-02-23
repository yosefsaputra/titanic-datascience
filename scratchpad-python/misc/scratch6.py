import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error

from datasets import datasets

from matplotlib import pyplot as plt

X, y = datasets.getErrorSinusoidal(100)
X = X[:, None]

# from sklearn.preprocessing import PolynomialFeatures
# X = PolynomialFeatures(2, include_bias=False).fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=20)

alphas = 10 ** np.linspace(6, -2, 100) * 0.5

# =================================================
questionName = 'Problem 1a'
print(questionName)

kfold = model_selection.KFold(n_splits=5)

lasso_mse = []
ridge_mse = []

lasso_coefficients = []
ridge_coefficients = []

for alpha in alphas:
    print('Alpha: %f' % alpha, end=' ')

    lasso_mse_alpha = []
    ridge_mse_alpha = []

    for train_index, test_index in kfold.split(X_train):
        lasso_model = linear_model.Lasso(alpha=alpha)
        ridge_model = linear_model.Ridge(alpha=alpha)

        lasso_model.fit(X_train[train_index], y_train[train_index])
        ridge_model.fit(X_train[train_index], y_train[train_index])

        lasso_predicted_y_train = lasso_model.predict(X_train[test_index])
        ridge_predicted_y_train = ridge_model.predict(X_train[test_index])

        lasso_mse_alpha.append(mean_squared_error(y_train[test_index], lasso_predicted_y_train))
        ridge_mse_alpha.append(mean_squared_error(y_train[test_index], ridge_predicted_y_train))

    lasso_mse.append(np.mean(lasso_mse_alpha))
    ridge_mse.append(np.mean(ridge_mse_alpha))


plt.figure()
plt.scatter(np.log(alphas), lasso_mse, label='lasso')
plt.scatter(np.log(alphas), ridge_mse, label='ridge')
plt.ylabel('Average of MSE of 5-fold cross validation')
plt.xlabel('log alpha')
plt.legend()

print('The best chosen lambda for lasso , based on the average MSE')


# =================================================
questionName = 'Problem 1b'
print(questionName)

plt.figure()
plt.scatter

plt.show()
