"""
Submission tests for Machine Learning Decision Tree Regressor

Author: Nathan Sprague
Version: 2/4/2021
"""

import unittest
import numpy as np
from decision_tree import DecisionTreeRegressor as DecisionTree


class MyTestCase(unittest.TestCase):

    def setUp(self):

        num = 20

        self.X = np.linspace(0, np.pi, num).reshape((num, 1))
        self.y_sin = np.sin(self.X).flatten()
        self.y_cos = np.cos(self.X).flatten()

        # 2 dimensional, where dim0 is irrelevant
        self.X2 = np.zeros((num, 2))
        self.X2[:, 0] = np.random.random(num)
        self.X2[:, 1] = np.linspace(0, 2 * np.pi, num)
        self.y2_sin = np.sin(self.X2[:, 1]).flatten()
        self.y2_cos = np.cos(self.X2[:, 1]).flatten()

    def test_depth_zero_predictions(self):
        tree = DecisionTree(0)
        tree.fit(self.X, self.y_sin)
        y_test = tree.predict(self.X)
        np.testing.assert_almost_equal(np.zeros(y_test.shape) +
                                       np.mean(self.y_sin), y_test)

        tree = DecisionTree(0)
        tree.fit(self.X, self.y_cos)
        y_test = tree.predict(self.X)
        np.testing.assert_almost_equal(np.zeros(y_test.shape) +
                                       np.mean(self.y_cos), y_test)

    def full_depth_on_training_helper(self, X, y, noise=0):
        tree = DecisionTree()
        tree.fit(X, y)
        y_test = tree.predict(X + (np.random.random(X.shape) - .5) * noise)
        np.testing.assert_array_almost_equal(y, y_test)

    def test_full_depth_on_training_points(self):
        self.full_depth_on_training_helper(self.X, self.y_sin)
        self.full_depth_on_training_helper(self.X, self.y_cos)
        self.full_depth_on_training_helper(self.X2, self.y2_sin)
        self.full_depth_on_training_helper(self.X2, self.y2_cos)

    def test_full_depth_on_randomized_training_points(self):
        self.full_depth_on_training_helper(self.X, self.y_sin, .001)
        self.full_depth_on_training_helper(self.X, self.y_cos, .001)
        self.full_depth_on_training_helper(self.X2, self.y2_sin, .001)
        self.full_depth_on_training_helper(self.X2, self.y2_cos, .001)

    def test_full_first_split_correct(self):
        tree = DecisionTree(1)
        tree.fit(self.X, self.y_sin)
        y_test_left = tree.predict(np.array([[.4]]))
        y_test_right = tree.predict(np.array([[.42]]))
        np.testing.assert_almost_equal(0.1630980, y_test_left)
        np.testing.assert_almost_equal(0.6811124, y_test_right)


if __name__ == '__main__':
    unittest.main()
