#!/usr/bin/env python3
"""Class represents a normal distribution."""


class Normal:
    """Class represents a normal distribution."""

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Normalize Distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum([(x - self.mean) ** 2 for x in data])
                                 / (len(data))) ** .5)

    def z_score(self, x):
        """Calculate z-score of x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate x-value of z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate value of pdf at x."""
        return (pow(Normal.e, ((x - self.mean) ** 2 /
                (-2 * self.stddev ** 2))) /
                (2 * Normal.pi * self.stddev ** 2) ** .5)

    def cdf(self, x):
        """Calculate value of cumulative distribution for x-value."""
        m = (x - self.mean) / (self.stddev * 2 ** .5)
        return (1 + (m - m ** 3 / 3 + m ** 5 / 10 - m ** 7
                     / 42 + m ** 9 / 216) * 2 / Normal.pi ** .5) / 2
