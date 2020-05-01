#!/usr/bin/env python3
"""Class constructor for poisson.py."""


class Poisson:
    """Poisson Class."""

    e = 2.7182818

    def __init__(self, data=None, lambtha=1.):
        """Intializig the class."""
        if data is None:
                if lambtha <= 0:
                    raise ValueError("lambtha must be a positive value")
                self.lambtha = lambtha
        else: 
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <  2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates probability mass at k."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return (pow(self.lambtha, k)
                * pow(Poisson.e, -1 * self.lambtha) / self.my_factorial(k))

    def cdf(self, k):
        """Calculates cumulative distribution at k."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return (sum([self.pmf(n) for n in range(k + 1)]))

    def my_factorial(self, m):
        """factorial of m."""
        if m == 1 or m == 0:
            return 1
        else:
            return m * self.my_factorial(m-1)
