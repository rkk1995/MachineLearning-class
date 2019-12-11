import sys
import numpy as np
from matplotlib import pyplot

sys.path.append('..')
from submission import SubmissionBase


def mapFeature(X1, X2, degree=6):

    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def plotDecisionBoundary(plotData, theta, X, y):
   
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)


class Grader(SubmissionBase):
    X = np.stack([np.ones(20),
                  np.exp(1) * np.sin(np.arange(1, 21)),
                  np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)

    y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(float)

    def __init__(self):
        part_names = ['Sigmoid Function',
                      'Logistic Regression Cost',
                      'Logistic Regression Gradient',
                      'Predict',
                      'Regularized Logistic Regression Cost',
                      'Regularized Logistic Regression Gradient']
        super().__init__('logistic-regression', part_names)

    def __iter__(self):
        for part_id in range(1, 7):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.X)
                elif part_id == 2:
                    res = func(np.array([0.25, 0.5, -0.5]), self.X, self.y)
                elif part_id == 3:
                    J, grad = func(np.array([0.25, 0.5, -0.5]), self.X, self.y)
                    res = grad
                elif part_id == 4:
                    res = func(np.array([0.25, 0.5, -0.5]), self.X)
                elif part_id == 5:
                    res = func(np.array([0.25, 0.5, -0.5]), self.X, self.y, 0.1)
                elif part_id == 6:
                    res = func(np.array([0.25, 0.5, -0.5]), self.X, self.y, 0.1)[1]
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0
