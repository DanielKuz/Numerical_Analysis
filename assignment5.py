"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from commons import shape1, shape3, shape5
import operator
from math import degrees, atan2, pi
from functionUtils import AbstractShape
import scipy.optimize as optimization
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random


class MyShape(AbstractShape):
    def __init__(self, clean_points):
        self.delta = clean_points
        self.x = [self.delta[i][0] for i in range(len(self.delta))]
        self.y = [self.delta[i][1] for i in range(len(self.delta))]

    def area(self):
        def Shoelace(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        area = Shoelace(self.x, self.y)
        return np.float32(area)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass



    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        def trapezoid(cont, i, n):
            points = cont(i * n)
            X = []
            Y = []
            for i in range(len(points)):
                X.append(points[i][0])
                Y.append(points[i][1])
            X += (X[0])
            Y += (Y[0])
            result = 0
            for i in range(1, len(X)):
                result += (X[i] - X[i - 1]) * 0.5 * (Y[i] + Y[i - 1])
            return abs(result)
        iteration = 1
        lim = 100
        res = trapezoid(contour, iteration, lim)
        while True:
            iteration += 1
            result = trapezoid(contour, iteration, lim)
            if (abs(result - res) / result) < maxerr:
                return np.float32(abs(result))
            res = result

    # def plot(self, points):
    #     # Plotting function
    #     x_lst = []
    #     y_lst = []
    #     for p in points:
    #         x_lst.append(p[0])
    #         y_lst.append(p[1])
    #     plt.scatter(x_lst, y_lst)
    #     plt.show()
    #
    # def bez_fit(self, points):
    #     # Copy paste from Assignment1
    #     n = len(points)
    #     sol_vec = [(0., 0.)] * (n - 1)
    #     C = 4 * np.identity(n)
    #     np.fill_diagonal(C[1:], 1)
    #     np.fill_diagonal(C[:, 1:], 1)
    #     C[0, 0] = 2
    #     C[n - 1, n - 1] = 7
    #     C[n - 1, n - 2] = 2
    #     sol_vec[0] = np.array(points[0]) + 2 * np.array(points[1])
    #     sol_vec[n - 2] = 8 * np.array(points[n - 2]) + np.array(points[n - 1])
    #     for i in range(1, n - 2):
    #         sol_vec[i] = 4 * np.array(points[i]) + 2 * np.array(points[i + 1])
    #     try:
    #         a, b, c = self.get_diagonal_vectors(C)
    #         A = self.thomas_algo(a, b, c, sol_vec)
    #     except:  # shouldn't happen but in case thomas acts weird
    #         A = np.linalg.solve(C, sol_vec)
    #     # Basic get bezier cubic as implemented in Assignment 1
    #     B = [2 * np.array(points[i]) - np.array(A[i]) for i in range(1, n - 1)]
    #     B += [np.array(A[-2]) + (2 * np.array(A[-1])) - (2 * np.array(B[-1]))]
    #     return A, B

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        def make_average_shape(shape):
            # Tried to implement this algorithm with knn algorithm but couldn't manage to find the needed functions
            def avg_point(points):
                n = len(points)
                xsum = 0
                ysum = 0
                for p in points:
                    xsum += p[0]
                    ysum += p[1]
                xcen = xsum / n
                ycen = ysum / n
                return xcen, ycen

            num_of_points = 1000
            xcenter, ycenter = avg_point(samples)
            # https://stackoverflow.com/questions/69402849/coordinates-clockwise-with-three-y-coordinates
            # Sorting the coordinates  clockwise
            shape.sort(
                key=lambda x_y: (-135 - degrees(atan2(*tuple(map(operator.sub, x_y, (xcenter, ycenter)))[::-1]))) % 360)
            points_avg = [(0., 0.)] * num_of_points
            group = int(len(shape) / num_of_points)
            # for each 10 nearest neighbors we will take the avg and we will iterate over 1000 as of now to get the
            # estimated shape
            for i in range(num_of_points):
                points = shape[i * group:(i + 1) * group]
                x, y = avg_point(points)
                points_avg[i] = (x, y)
            return points_avg

        n = 100000
        # sampling
        samples = [sample() for i in range(n)]
        # getting the estimated shape
        clean_samples = make_average_shape(samples)
        return MyShape(clean_samples)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm

class TestAssignment5(unittest.TestCase):
    def test_are(self):
        ass5 = Assignment5()
        T = time.time()
        ar = ass5.area(shape3().contour, 0.001)
        print(ar)
        print(shape5().area())
        print(shape5().area()-ar)

    def test_return(self):
        radius = 100
        circ = noisy_circle(cx=1, cy=1, radius=radius, noise=0)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        a = pi * (radius ** 2)
        print(abs(a - shape.area()) / a)
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_shape1(self):
        shape11 = shape1()
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape11.sample, maxtime=20)
        T = time.time() - T
        print(abs(shape11.area() - shape.area()) / shape11.area())
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 20)

    def test_shape3(self):
        shape11 = shape3()
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape11.sample, maxtime=20)
        T = time.time() - T
        print(abs(shape11.area() - shape.area()) / shape11.area())
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 20)

    def test_shape5(self):
        shape11 = shape5()
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape11.sample, maxtime=20)
        T = time.time() - T
        print(abs(shape11.area() - shape.area()) / shape11.area())
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 20)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
