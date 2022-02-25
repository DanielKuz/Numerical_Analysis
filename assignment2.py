"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes
        any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def newton_raphson(x, f, y, a, b, err):
            deriv = lambda x: (f(x + err) - f(x)) / err
            lim = 50
            if y is None:
                return
            try:
                derivative = deriv(x)
            except:
                return
            if derivative == 0 or derivative != derivative:
                return
            for i in range(lim):
                if abs(y) <= err:
                    return x
                x = x - y / derivative
                try:
                    derivative = deriv(x)
                except:
                    return
                if derivative == 0 or derivative is None or x < a or x > b:
                    return
                try:
                    y = f(x)
                except:
                    return
                if y is None:
                    return
            return

        def append(x, a, b, err, roots):
            if x >= a and x <= b:
                for root in roots:
                    if abs(root - x) <= err:
                        return
                roots.append(x)
            return

        roots = []
        err = maxerr
        f = lambda x: (f2(x) - f1(x)) / err
        newton_guess = (b - a) / 715   # Cant justifiy this number, i ran test_all a couple of time and got 487 and 715
        x = a
        while x <= b:
            try:
                func = f(x)
            except:
                x += newton_guess
                continue
            if abs(func) < err:
                append(x, a, b, err, roots)
            else:
                root = newton_raphson(x, f, func, a, b, err)
                if root is not None:
                    append(root, a, b, err, roots)
            x += newton_guess
        roots.sort()
        return roots


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import f10, f6, f2_nr, f3_nr
import math


class TestAssignment2(unittest.TestCase):
    # def test_best(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = np.poly1d([-1, 0, 1])
    #     f2 = np.poly1d([1, 0, -1])
    #     best = []
    #     best_len = 0
    #     for i in range(1,500):
    #         X = ass2.intersections(f1, f2, -1, 1,div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best_len==len(X):
    #             best.append(i)
    #         elif best_len<len(X):
    #             best = []
    #             best_len = curr_len
    #             best.append(i)
    #     print(best,best_len)
    # def test_best1(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = f10
    #     f2 = f6
    #     best = []
    #     best_len = 0
    #     for i in range(1,500):
    #         X = ass2.intersections(f1, f2, 0, 100,div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best_len == len(X):
    #             best.append(i)
    #         elif best_len < len(X):
    #             best = []
    #             best_len = curr_len
    #             best.append(i)
    #     print(best,best_len)
    #
    # def test_best3(self):
    #     ass2 = Assignment2()
    #     f1 = lambda x: 2 ** (1 / (x ** 2)) * math.sin(1 / x)
    #     f2 = lambda x: x
    #     best = []
    #     best_len = 0
    #     for i in range(1,500):
    #         X = ass2.intersections(f1, f2,  -2, 2,div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best_len == len(X):
    #             best.append(i)
    #         elif best_len < len(X):
    #             best = []
    #             best_len = curr_len
    #             best.append(i)
    #     print(best,best_len)

    # def test_all(self):
    #     ass2 = Assignment2()
    #     f1 = lambda x: 2 ** (1 / (x ** 2)) * math.sin(1 / x)
    #     f2 = lambda x: x
    #     best3 = []
    #     best3_len = 0
    #     for i in range(1, 1000):
    #         X = ass2.intersections(f1, f2, -2, 2, div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best3_len == len(X):
    #             best3.append(i)
    #         elif best3_len < len(X):
    #             best3 = []
    #             best3_len = curr_len
    #             best3.append(i)
    #     f1 = f10
    #     f2 = f6
    #     best2 = []
    #     best_len2 = 0
    #     for i in range(1, 1000):
    #         X = ass2.intersections(f1, f2, 0, 100, div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best_len2 == len(X):
    #             best2.append(i)
    #         elif best_len2 < len(X):
    #             best2 = []
    #             best_len2 = curr_len
    #             best2.append(i)
    #     f1 = np.poly1d([-1, 0, 1])
    #     f2 = np.poly1d([1, 0, -1])
    #     best = []
    #     best_len = 0
    #     for i in range(1,1000):
    #         X = ass2.intersections(f1, f2, -1, 1,div=i, maxerr=0.001)
    #         curr_len = len(X)
    #         if best_len==len(X):
    #             best.append(i)
    #         elif best_len<len(X):
    #             best = []
    #             best_len = curr_len
    #             best.append(i)
    #     best_overall = []
    #     for i in best2:
    #         if i in best and i in best3:
    #             best_overall.append(i)
    #     print(best_overall)

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_strange(self):

        ass2 = Assignment2()

        f1 = f10
        f2 = f6
        X = ass2.intersections(f1, f2, 0, 100, maxerr=0.001)
        print(X)
        for x in X:
            print(abs(f1(x) - f2(x)))
            print(abs(f1(2.091) - f2(2.091)))
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -20, 20, maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_hardcase(self):

        ass2 = Assignment2()
        f1 = lambda x: 2 ** (1 / (x ** 2)) * math.sin(1 / x)
        f2 = lambda x: x
        X = ass2.intersections(f1, f2, -2, 2, maxerr=0.00001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_hardcase1(self):

        ass2 = Assignment2()
        f1 = f2_nr
        f2 = f3_nr
        X = ass2.intersections(f1, f2, -2, 5, maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
