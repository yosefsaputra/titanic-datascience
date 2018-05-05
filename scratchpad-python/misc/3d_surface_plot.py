'''
Generating plots of A+B+C=1 and ABC=0.024 for Programming with Molecules HW4
'''

import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np


def submission(x, y):
    z = 1 - x - y
    return z


def multiplication(x, y):
    z = 0.024 / x / y
    return z


x = y = np.arange(0.05, 1.0, 0.01)
X, Y = np.meshgrid(x, y)

zs_mul = np.array([multiplication(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
zs_sub = np.array([submission(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])

data = [
    go.Surface(
        x=x,
        y=y,
        z=zs_sub.reshape(X.shape),
        name='a + b + c'
    ),
    go.Surface(
        x=x,
        y=y,
        z=zs_mul.reshape(X.shape),
        name='a . b . c'
    )
]
layout = go.Layout(
    autosize=True,
    scene=dict(
        xaxis=dict(
            nticks=4, range=[0, 1], title='a'),
        yaxis=dict(
            nticks=4, range=[0, 1], title='b'),
        zaxis=dict(
            nticks=4, range=[0, 1], title='c'), ),
)


fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
