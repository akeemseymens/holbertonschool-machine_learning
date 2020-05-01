#!/usr/bin/env python
"""Exponential distribution."""


class Exponential:
    """Intializig the class."""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initialize exponential distribution."""
        if data is None and isinstance(lambtha, (float, int)):
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculate the probability density given time period x."""
        if x is None or x < 0:
            return 0
        return (self.lambtha * pow(Exponential.e, (-1 * self.lambtha) * x))

    def cdf(self, x):
        """Calculate cumulative distribution for a given time period x."""
        if x is None or x < 0:
            return 0
        return (1 -(pow(Exponential.e , (-1 * self.lambtha) * x)))
