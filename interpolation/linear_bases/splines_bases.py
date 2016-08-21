import numpy

import sympy

x = numpy.linspace(0,1,10)

x.searchsorted(0.1)

def B(k,i,x,u):
    i0 = numpy.searchsorted(x,u)
    a = x[i0]
    b = x[i0+1]
    v = u
