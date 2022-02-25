"""
In this assignment you should interpolate the given function.
"""
import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def get_diagonal_vectors(self, A):
        a = A.diagonal(-1)
        b = A.diagonal()
        c = A.diagonal(1)
        return a, b, c

    # Thomas algorithm using numpy diagonal
    # we can use this as we know the coef matrix will be tridiagonal
    def thomas_algo(self, a, b, c, d):
        nf = len(d)  # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
        for it in range(1, nf):
            mc = ac[it - 1] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]
        xc = dc
        xc[-1] = dc[-1] / bc[-1]
        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
        return xc

    def get_bezier_coef(self, points):
        n = len(points) - 1
        # coef matrix
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2
        # point vec
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]
        # Thomas to solve tridiagonal matrix, as seen in NA6
        a_diag, b_diag, c_diag = self.get_diagonal_vectors(C)
        A = self.thomas_algo(a_diag, b_diag, c_diag, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2
        return A, B

    # returns the general Bezier cubic formula given 4 control points
    def get_cubic(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                          2) * c + np.power(
            t, 3) * d

    # return one cubic curve for each consecutive points
    def get_bezier_cubic(self, points):
        A, B = self.get_bezier_coef(points)  # TODO: Should be changed to dict
        return [
            self.get_cubic(points[i], A[i], B[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        x_points = np.linspace(a, b, n)
        points = [np.array([x_points[i], f(x_points[i])]) for i in range(0, n)]
        points = np.array(points)
        bezier = self.get_bezier_cubic(points)

        def bez_return_func(x_points, bez, n):
            def g(x):
                if x <= x_points[0]:
                    return bez[0](x)[1]
                if x >= x_points[n - 1]:
                    return bez[n - 1](x)[1]
                for j in range(0, n - 1):
                    if (x >= x_points[j]) and (x <= x_points[j + 1]):
                        return bez[j]((x - x_points[j]) / (x_points[j + 1] - x_points[j]))[1]

            return g

        result = bez_return_func(x_points, bezier, n)
        return result


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
