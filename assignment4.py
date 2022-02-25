"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

import numpy as np
import time
import random
import math
import torch


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        self.m = np.array([])

    def LUsolver(self,A, b):
        def LU_decomposition(A):
            """Perform LU decomposition of a square matrix (numpy array) using the Doolittle factorization."""
            L = np.zeros_like(A, dtype=float)
            U = np.zeros_like(A, dtype=float)
            N = np.size(A, 0)

            for k in range(N):
                L[k, k] = 1
                U[k, k] = A[k, k] - np.dot(L[k, :k], U[:k, k])
                for j in range(k + 1, N):
                    U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
                for i in range(k + 1, N):
                    L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

            return L, U

        L, U = LU_decomposition(A)
        l = len(b)
        y = np.zeros(l)
        y[0] = b[0] / L[0, 0]
        for i in range(1, l):
            y[i] = (b[i] - np.dot(L[i, : i], y[: i])) / L[i, i]

        l = len(y)
        x = np.zeros(l)
        x[l - 1] = y[l - 1] / U[l - 1, l - 1]
        for i in range(2, l + 1, 1):
            x[-i] = (y[-i] - np.dot(U[-i, -i:], x[-i:])) / U[-i, -i]

        return x
    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float, p=2) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        def fill_mat(x_pow):
            return np.array([[self.m[i,j]+x_pow[i+j] for i in range(d+1)] for j in range(d+1)])
        T= time.time()
        f((a+b))
        T1 = time.time()
        time_per_sample = T1-T
        if time_per_sample > maxtime:
            return
        elif time_per_sample == 0:
            time_per_sample = 0.0001
        elif time_per_sample <= 0.0005:
            time_per_sample = 0.0001
        else:
            time_per_sample = time_per_sample * 1.1
        samp_range = int((maxtime-1.5)/time_per_sample)
        x_pow = np.array([])
        self.m = np.zeros(shape=(d+1,d+1))
        x_points = np.linspace(a, b, samp_range)
        y_points = [f(x) for x in x_points]
        for i in range(2 * d + 1):
            s = (x_points ** i).sum()
            x_pow = np.append(x_pow, s)
        self.m = fill_mat(x_pow)
        b_vec = np.array([])
        for i in range(d + 1):
            b_vec = np.append(b_vec, np.dot(x_points ** i, y_points))
        sol = self.LUsolver(self.m, b_vec)
        poly_ans = np.poly1d(np.flip(sol))
        return poly_ans
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import f9_noise,f9


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        print(T)
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        print(T)
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = DELAYED(0.5)(NOISY(0.01)(poly(1, 1, 1)))
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=3, maxtime=5)
        T = time.time() - T
        mse = 0
        print(T)
        f = poly(1, 1, 1)
        for x in np.linspace(0, 1, 1000):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)

    def test_err1(self):
        f = poly(-9, -9, 1)
        nf = DELAYED(0.5)(NOISY(1)(f))
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=3, maxtime=10)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print("mse: ", mse)
    def test_err3(self):
        f = np.sin
        nf = DELAYED(0.5)(NOISY(1)(f))
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=3, maxtime=20)
        T = time.time() - T
        mse = 0
        f = np.sin
        for x in np.linspace(0, 1, 1000):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print("mse: ", mse)
    def test_err4(self):
        f = DELAYED(0.5)(NOISY(1)(f9))
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=f, a=5, b=10, d=10, maxtime=20)
        T = time.time() - T
        print(T)
        mse = 0
        f = f9
        for x in np.linspace(5, 10, 1000):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print("mse: ", mse)


if __name__ == "__main__":
    unittest.main()
