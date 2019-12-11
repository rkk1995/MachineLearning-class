import sys
import numpy as np
from scipy import optimize
from matplotlib import pyplot

sys.path.append('..')
from submission import SubmissionBase


def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
   
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x


def featureNormalize(X):

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)


class Grader(SubmissionBase):
    # Random test cases
    X = np.vstack([np.ones(10),
                   np.sin(np.arange(1, 15, 1.5)),
                   np.cos(np.arange(1, 15, 1.5))]).T
    y = np.sin(np.arange(1, 31, 3))
    Xval = np.vstack([np.ones(10),
                      np.sin(np.arange(0, 14, 1.5)),
                      np.cos(np.arange(0, 14, 1.5))]).T
    yval = np.sin(np.arange(1, 11))

    def __init__(self):
        part_names = ['Regularized Linear Regression Cost Function',
                      'Regularized Linear Regression Gradient',
                      'Learning Curve',
                      'Polynomial Feature Mapping',
                      'Validation Curve']
        super().__init__('regularized-linear-regression-and-bias-variance', part_names)

    def __iter__(self):
        for part_id in range(1, 6):
            try:
                func = self.functions[part_id]
                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.X, self.y, np.array([0.1, 0.2, 0.3]), 0.5)
                elif part_id == 2:
                    theta = np.array([0.1, 0.2, 0.3])
                    res = func(self.X, self.y, theta, 0.5)[1]
                elif part_id == 3:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval, 1)).tolist()
                elif part_id == 4:
                    res = func(self.X[1, :].reshape(-1, 1), 8)
                elif part_id == 5:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval)).tolist()
                else:
                    raise KeyError
            except KeyError:
                yield part_id, 0
            yield part_id, res

