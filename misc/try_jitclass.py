import numpy as np
from interpolation.splines.filter_cubic import filter_coeffs_3d
from interpolation.splines.eval_cubic import eval_cubic_spline_3
from numba import jitclass, njit, float64, int64
from collections import OrderedDict
from numpy import array


spec = OrderedDict()
spec['a'] = float64[:]
spec['b'] = float64[:]
spec['orders'] = int64[:]
spec['d'] = int64
spec['coeffs'] = float64[:,:,:]

@jitclass(spec)
class Spline3D:

    # a = None
    # b = None
    # orders = None
    # d = None
    # __coeffs__ = None

    def __init__(self,a,b,orders,values):

        self.a = a
        self.b = b
        self.orders = orders
        dinv = (self.a - self.b) / self.orders
        self.coeffs = filter_coeffs_3d(dinv,values)

    def evaluate(self, point):
        return eval_cubic_spline_3(self.a, self.b, self.orders, self.coeffs, point)


a = np.array([0.0,0.0, 0.0])
b = np.array([1.0,1.0,1.0])
orders = np.array([50,50,50])
values = np.random.random(orders)

sp = Spline3D(a,b,orders,values)
point = np.array([0.5,0.5, 0.5])
res = sp.evaluate(point)


# test performances on many points vs. "low-level" routines
N = 10**6
points = np.random.random( (N,3) )
coeffs = sp.coeffs


@njit
def repeat_eval(sp, points, out):
    N = points.shape[0]
    vals = np.zeros(3)
    for n in range(N):
        pp = points[n,:]
        out[n] = sp.evaluate( pp )
    return vals

@njit
def repeat_eval_noclass(a,b,orders,coeffs, points,out):
    N = points.shape[0]
    vals = np.zeros(3)
    for n in range(N):
        pp = points[n,:]
        out[n] = eval_cubic_spline_3(a, b, orders, coeffs, pp)
    return vals



out = np.zeros(N)

repeat_eval(sp, points, out) # warmup
repeat_eval_noclass(a, b, orders, coeffs, points,out)


import time
t1 = time.time()
tot = repeat_eval(sp, points,out)
t2 = time.time()
print("JIT class (repeated call): {}".format(t2-t1))



import time
t1 = time.time()
tot = repeat_eval_noclass(a, b, orders, coeffs, points,out)
t2 = time.time()
print("No class (repeated call): {}".format(t2-t1))




# optimized call:
from interpolation.splines.eval_cubic import vec_eval_cubic_spline_3
out2 = np.zeros(N)
# warmup
vec_eval_cubic_spline_3(sp.a, sp.b, sp.orders, sp.coeffs, points, out2)
t1 = time.time()
vec_eval_cubic_spline_3(a, b, orders, coeffs, points, out2)
t2 = time.time()
print("Vectorized version: {}".format(t2-t1))


# compare with scipy (linear)
# probably not the best way to use it !
from scipy.interpolate import RegularGridInterpolator

pp = [np.linspace(a[i],b[i],orders[i]) for i in range(3)]
regint = RegularGridInterpolator(pp, values)
t1 = time.time()
out_scipy = regint(points)
t2 = time.time()

print("Scipy (linear): {}".format(t2-t1))

assert(abs(out-out2).max()<1e-10)
