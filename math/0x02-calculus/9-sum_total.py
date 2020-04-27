#!/usr/bin/env python3
'''that calculates \sum_{i=1}^{n} i^2:'''


def summation_i_squared(n):
   return sum(i**2 for i in range(1, n+1))