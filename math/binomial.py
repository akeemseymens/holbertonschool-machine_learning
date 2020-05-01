#!/usr/bin/env python3
"""Class represents a binomial distribution."""


class Binomial:
    """Intializing the class."""

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            vari = sum([(m - mean) ** 2 for m in data]) / len(data)
            self.p = -1 * (vari / mean - 1)
            n = mean / self.p
            self.n = round(n)
            self.p *= n / self.n

    def pmf(self, k):
        """Calculate value of PMF for given number k."""
        if type(k) is not int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        return (self.my_factorial(self.n) / self.my_factorial(k) / self.my_factorial(self.n - k)
                * self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculate value CDF for given number k."""
        c_prob = 0

        if type(k) is not int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        for m in range(0, k + 1):
            c_prob += self.pmf(m)
        return c_prob

    def my_factorial(self, m):
        """Factorializing of m."""
        if m == 1 or m == 0:
            return 1
        else:
            return m * self.my_factorial(m-1)
