from interpolation.splines.eval_cubic_numba import vec_eval_cubic_spline_3
from interpolation.splines.filter_cubic import filter_coeffs
from interpolation.splines.multilinear_numba import multilinear_interpolation

from dolo.numeric.misc import mlinspace

import numpy

K = 50
d = 3
N = 10**6
N = 100
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


from numba import cuda
from interpolation.splines.eval_cubic_cuda import vec_eval_cubic_spline_3 as original
jitted = cuda.jit(original)

res_cuda = numpy.zeros_like(res)

d_res_cuda = cuda.to_device(res_cuda)
jitted[(N,1,1)](a,b,orders,C,X,d_res_cuda)
res_cuda = d_res_cuda.copy_to_host()

cuda.synchronize()
t4 = time.time()
d_res_cuda = cuda.to_device(res_cuda)
jitted[(N,1,1)](a,b,orders,C,X,d_res_cuda)
res_cuda = d_res_cuda.copy_to_host()
cuda.synchronize()
t5 = time.time()

assert( abs(res_cuda - res).max()<1e-8 )

print("CUDA: {}" .format(t5-t4))
