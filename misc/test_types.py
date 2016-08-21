
import numba
from numba import jit


@jit
def fun(x):
    "This is the doc"
    return x[0]+x[1]

arg = (2,3)
fun(arg)


sig = fun.signatures[0]

sig

from numba.types.containers.
sig.__class__

fun((2.1,2.0))
fun((2.1,2))

from numba.types import int64

fun.get_overload(sig=(int64,int64))



sig = fun.nopython_signatures[0][0]

sig

fun((2,1))


tt = fun.typeof_pyval(((3,2),4))


from numba import jitclass

@jitclass(spec=[('data', int64)])
class Test:

    def __init__(self, data):
        self.data = data

    def test_me(self, i):
        return self.data + i


cls = Test(4)
cls.test_me(3)

import numba.types
from numpy import zeros, array
from numba import float64

tt = fun.typeof_pyval(ar)
tt

@jitclass(spec=[('data', float64[:,:])])
class Test:

    def __init__(self, data):
        self.data = data

    def test_me(self, i):
        return self.data[0,0] + i

ar = array([[0.0]])
tt
tt
ar
tt
cls = Test(ar)
cls.test_me(1.0)
