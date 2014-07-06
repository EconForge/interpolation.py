from __future__ import division

d = 1
k = 100
N_tries = 100
ncpus = 1

N = 128*100000


import os
os.environ['OMP_NUM_THREADS'] = str( ncpus )

import numpy as np

from numpy import array, sin
from dolo.numeric.interpolation.multilinear import mlinspace


smin = array([0.0]*d,dtype=float)
smax = array([1.0]*d,dtype=float)
orders = array([k]*d,dtype=int)

svec = mlinspace( [0]*d, [1]*d, orders)

f = lambda x:  np.atleast_2d( (sin(x*10)).sum(axis=0) )


values = f(svec)


points = np.random.random( (d,N))
points = np.maximum(points, -0.1) #0.001)
points = np.minimum(points, 1.1) #0.999)


#points = np.atleast_2d( np.linspace(-2,2,N).T )
#points = points.copy()

import time


from interpolation.splines_numba import *
from interpolation.splines_filter_numba import *

import time

t0 = time.time()
coeffs = filter_coeffs(smin,smax,orders,values)

vals = zeros( points.shape[1] )


t1 = time.time()

#print("Elapsed : {}".format(t1-t0))

ev = eval_UBspline(smin, smax, orders, coeffs, points)


t2 = time.time()

for i in range(N_tries):
  ev = eval_UBspline(smin, smax, orders, coeffs, points)

t3 = time.time()

#print(t2-t1)

print("Numba")
print(t3-t2)
print("Error : {}".format( (ev-f(points)).max()  ))



from dolo.numeric.interpolation.splines import MultivariateSplines as MultivariateSplines
sppp = MultivariateSplines(smin,smax,orders)
print(values.shape)
print(svec.shape)

sppp.set_values(values)

import numpy
svec = numpy.ascontiguousarray(svec)
b = sppp(points)

ss = time.time()
for i in range(N_tries):
    b = sppp(points)
tt = time.time()
print("Cython")
print(tt-ss)


