from interpolation.splines.eval_cubic_numba import vec_eval_cubic_spline_3
from interpolation.splines.filter_cubic import filter_coeffs
from interpolation.splines.multilinear_numba import multilinear_interpolation

from interpolation.splines.misc import mlinspace

import numpy

K = 50
d = 3
N = 10**6
T = 10 # number of repetitions
dtype = numpy.float64

a = numpy.array([0.0,0.0,0.0],dtype=dtype)
b = numpy.array([1.0,1.0,1.0],dtype=dtype)
orders = numpy.array([K,K,K],dtype=numpy.int32)

V = numpy.random.random(orders)
V = V.astype(dtype)
C = filter_coeffs(a,b,orders,V)
C = C.astype(dtype)

X = numpy.random.random((N,3))*0.5+0.5
X = X.astype(dtype)
res = numpy.zeros(N,dtype=dtype)
res2 = res.copy()

vec_eval_cubic_spline_3(a,b,orders,C,X,res)

multilinear_interpolation(a,b,orders,V,X,res)



import time

t1 = time.time()
for t in range(T):
    vec_eval_cubic_spline_3(a,b,orders,C,X,res)
t2 = time.time()
for t in range(T):
    multilinear_interpolation(a,b,orders,V,X,res2)
t3 = time.time()

print("Cubic: {}".format((t2-t1)/T))
print("Linear: {}".format((t3-t2)/T))


from numba import cuda
from interpolation.splines.eval_cubic_cuda import vec_eval_cubic_spline_3 as original
from interpolation.splines.eval_cubic_cuda import Ad,dAd

Ad = Ad.astype(dtype)
dAd = dAd.astype(dtype)

jitted = cuda.jit(original)

# warmup
res_cuda = numpy.zeros_like(res)
jitted[(N,1)](a,b,orders,C,X,res_cuda,Ad,dAd)


cuda.synchronize()
t4 = time.time()
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_orders = cuda.to_device(orders)
d_C = cuda.to_device(C)
d_Ad = cuda.to_device(Ad)
d_dAd = cuda.to_device(dAd)
d_X = cuda.to_device(X)
d_res_cuda = cuda.to_device(res_cuda)
for t in range(T):
    jitted[(N,1)](d_a,d_b,d_orders,d_C,d_X,d_res_cuda,d_Ad,d_dAd)
res_cuda = d_res_cuda.copy_to_host()
cuda.synchronize()
t5 = time.time()


print("CUDA: {}" .format((t5-t4)/T))

assert( abs(res_cuda - res).max()<1e-8 )
