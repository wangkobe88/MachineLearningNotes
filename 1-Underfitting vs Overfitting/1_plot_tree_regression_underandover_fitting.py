"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""
print(__doc__)

import numpy as np

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80*5, 1), axis=0)
print len(X)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16*5))
# Fit regression model
from sklearn.tree import DecisionTreeRegressor

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 4))
for i in range(0,6):
    ax = plt.subplot(2, 3, i+1)
    plt.setp(ax, xticks=(), yticks=())
    clf_1 = DecisionTreeRegressor(max_depth=(i+1))
    clf_1.fit(X, y)
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = clf_1.predict(X_test)
    plt.scatter(X, y, c="k", label="data")
    plt.plot(X_test, y_1, c="r", label="max_depth"+str(i+1), linewidth=3)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
plt.show()
