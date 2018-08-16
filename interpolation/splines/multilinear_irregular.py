# from .multilinear_irregular_numba import *

import numba
#
@numba.njit
def mysum(a):
    return a

from numba.extending import overload
@overload(mysum)
def mysum2(a,b):
    def ff(a,b):
        return a + b
    return ff

mysum(1,2)
#
# @numba.njit
# def test():
#     return max(1,2) + max(1,2,3)
#
# print(test())
