from interpolation.splines.eval_cubic_splines_numba import vec_eval_cubic_spline_3, vec_eval_cubic_spline_3_vec, vec_eval_cubic_spline_3_kernel
from dolo.numeric.misc import mlinspace

import numpy

K = 50
d = 3
N = 10**6

a = numpy.array([0.0,0.0,0.0])
b = numpy.array([1.0,1.0,1.0])
orders = numpy.array([K,K,K],dtype=int)
C = numpy.random.random((K,K,K))

X = numpy.random.random((N,3))
res = numpy.zeros(N)

vec_eval_cubic_spline_3(a,b,orders,C,X,res)


import time

t1 = time.time()
vec_eval_cubic_spline_3(a,b,orders,C,X,res)
t2 = time.time()

print("Elapsed: {}".format(t2-t1))
