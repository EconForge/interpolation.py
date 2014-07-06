from __future__ import division

d = 2
k = 50
N_tries = 10
ncpus = 4



import os
os.environ['OMP_NUM_THREADS'] = str( ncpus )

import numpy as np

from dolo.numeric.interpolation.multilinear import mlinspace


#import pyximport; pyximport.install()
#from dolo.numeric.interpolation.splines_cython import *





smin = [0]*d
smax = [1]*d
orders = [k]*d

svec = mlinspace( [0]*d, [1]*d, orders)

f = lambda x:  np.atleast_2d( (x**2).sum(axis=0) )
#f = lambda x:  np.atleast_2d( x**2 )


values = f(svec)

N = 100000

points = np.random.random( (d,N))
points = np.maximum(points, 0.001)
points = np.minimum(points, 0.999)


#points = np.atleast_2d( np.linspace(-2,2,N).T )
#points = points.copy()

import time


#spline = USpline(smin,smax,orders,values.reshape(orders))



#t = time.time()
#for i in range(10):
#    interp1 = eval_UBspline(smin,smax,orders,spline.coefs,points)
#s = time.time()


#start_pdiff = time.time()
#for i in range(10):
#    [ff0,dff0] = np.array( eval_UBspline(smin,smax,orders,spline.coefs,points,diff=True) )
#end_pdiff = time.time()

dtype = np.float64
#dtype = np.float32

from dolo.numeric.interpolation.splines import MultivariateSplines as MultivariateSplinesNew
sppp = MultivariateSplinesNew(np.array(smin,dtype=dtype),np.array(smax,dtype=dtype), np.array(orders,dtype=int), dtype=dtype)
sppp.set_values(values)

elapsed = []
for i in range(10):
    tdpp = time.time()
    b = sppp(points)
    sdpp = time.time()
    elapsed.append(sdpp - tdpp)

print(b.dtype)
print('Cubic splines')
print('-------------')
print('dimensions: {}'.format(d))
print('orders: {}'.format(orders[:d]))
print('points: {}'.format(N))

print('approximation error : {}'.format( abs(b - f(points) ).max()) )

print('')
print('timing')
print('------')

#print('python (evaluation + extrapolation)                  : {} s \t: {} 10^6 ev/s'.format( (s-t)/N_tries,  int(N*N_tries/(s-t)/1000000)))
#print('python (evaluation + derivatives [no extrap])        : {} s \t: {} 10^6 ev/s'.format( (end_pdiff - start_pdiff)/N_tries, int(N*N_tries/(end_pdiff - start_pdiff)/1000000)) )
total = sum(elapsed[1:])
print('python (evaluation [object oriented])                : {} s \t: {:.3f} 10^6 ev/s'.format( total/(N_tries-1), N*(N_tries-1)/(total/1000000)) )


