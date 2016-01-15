from interpolation.splines.eval_cubic_numba import vec_eval_cubic_spline_3
from interpolation.splines.filter_cubic import filter_coeffs
from interpolation.splines.multilinear_numba import multilinear_interpolation

from interpolation.splines.misc import mlinspace

import numpy

K = 50
d = 3
N = 10**6
# N = 100

a = numpy.array([0.0,0.0,0.0])
b = numpy.array([1.0,1.0,1.0])
orders = numpy.array([K,K,K],dtype=int)

V = numpy.random.random(orders)
C = filter_coeffs(a,b,orders,V)


X = numpy.random.random((N,3))
res = numpy.zeros(N)
res2 = res.copy()

vec_eval_cubic_spline_3(a,b,orders,C,X,res)

multilinear_interpolation(a,b,orders,V,X,res)



import time

t1 = time.time()
vec_eval_cubic_spline_3(a,b,orders,C,X,res)
t2 = time.time()
multilinear_interpolation(a,b,orders,V,X,res2)
t3 = time.time()

print("Cubic: {}".format(t2-t1))
print("Linear: {}".format(t3-t2))


print(abs(res-res2).max())

