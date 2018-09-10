import logistic_regression as lr
import numpy as np
from logistic_regression import logistic_regression_GD

fileX = 'logistic_x.txt'
fileY = 'logistic_y.txt'


def load_data(fileX, fileY):
    X = np.loadtxt(fileX)
    y = np.loadtxt(fileY)
    ones = np.ones((99, 1))
    Xsplit = np.split(X, indices_or_sections=[1], axis=1)
    # Pack the intercept coordinates into X so we can calculate the 
    # intercept for the logistic regression.
    X = np.concatenate([ones, Xsplit[0], Xsplit[1]], axis=1)
    return (X, y)


def regression(epsilon=lr.EPSILON, max_iters=lr.MAX_ITERS):
    X, y = load_data(fileX, fileY)
    return lr.logistic_regression(X, y, epsilon, max_iters)


def regression_GD(epsilon=lr.EPSILON, max_iters=lr.MAX_ITERS):
    X, y = load_data(fileX, fileY)
    return lr.logistic_regression_GD(X, y, epsilon, max_iters)


def main():
    theta, cost = regression()

    print('theta = {}'.format(theta))
    print('cost = {}'.format(cost))


if __name__ == '__main__':
    regression()
    # regression_GD()
    main()
