### tensor product of many many vectors
from dolo.misc.timing import timeit

import time
from contextlib import contextmanager


import numpy as np
@contextmanager
def timeit(msg):
    t1 = time.time()
    yield
    t2 = time.time()
    print('{}: {:.4f} s'.format(msg, t2-t1))


import numpy
from numpy import empty
from numba import jit

@jit
def tensor_product(A,B,C):
    N_A = A.shape[0]
    N_B = B.shape[0]
    N_C = C.shape[0]
    OUT = empty((N_A,N_B,N_C))
    for i in range(N_A):
        for j in range(N_B):
            for k in range(N_C):
                OUT[i,j,k] = A[i]*B[j]*C[k]
    return OUT

import numpy

def tensor_product_bcast_1(A,B,C):
    return A[:,None,None]*B[None,:,None]*C[None,None,:]

def tensor_product_bcast_2(A,B,C):
    return (A[:,None]*B[None,:])[:,:,None]*C[None,None,:]

def tensor_product_kron(A,B,C):
    N = A.shape[0]
    return numpy.kron(numpy.kron(A,B),C).reshape((N,N,N))

def tensor_product_outer(A,B,C):
    N = A.shape[0]
    return numpy.outer(numpy.outer(A,B),C).reshape((N,N,N))

N = 500
A = numpy.random.random(N)
B = numpy.random.random(N)
C = numpy.random.random(N)

# warmup
oo = tensor_product(A,B,C)
oo1 = tensor_product_bcast_1(A,B,C)
oo2 = tensor_product_bcast_2(A,B,C)
oo3 = tensor_product_kron(A,B,C)

print("- Simple tensor product- ")
with timeit("jit"):
    for i in range(10): oo = tensor_product(A,B,C)
with timeit("bcast"):
    for i in range(10):  oo1 = tensor_product_bcast_1(A,B,C)
with timeit("bcast (grouped)"):
    for i in range(10):  oo2 = tensor_product_bcast_2(A,B,C)
with timeit("kronecker"):
    for i in range(10):  oo3 = tensor_product_kron(A,B,C)
with timeit("outer"):
    for i in range(10):  oo4 = tensor_product_outer(A,B,C)
with timeit("einsum"):
    for i in range(10):  oo5 = numpy.einsum('i,j,k->ijk',A,B,C)

assert(abs(oo1-oo).max()<1e-16)
assert(abs(oo2-oo).max()<1e-16)
assert(abs(oo3-oo).max()<1e-16)
assert(abs(oo4-oo).max()<1e-16)
assert(abs(oo5-oo).max()<1e-16)

### vectorized tensor product of many many vectors

import numpy
from numpy import empty
from numba import jit

@jit
def vec_tensor_product(A,B,C):
    N = A.shape[0]
    N_A = A.shape[1]
    N_B = B.shape[1]
    N_C = C.shape[1]
    OUT = empty((N,N_A,N_B,N_C))
    for n in range(N):
        for i in range(N_A):
            for j in range(N_B):
                for k in range(N_C):
                    OUT[n,i,j,k] = A[n,i]*B[n,j]*C[n,k]
    return OUT

@jit
def vec_tensor_product_2(A,B,C):
    N = A.shape[0]
    N_A = A.shape[1]
    N_B = B.shape[1]
    N_C = C.shape[1]
    OUT = empty((N,N_A,N_B,N_C))
    for i in range(N_A):
        for j in range(N_B):
            for k in range(N_C):
                for n in range(N):
                    OUT[n,i,j,k] = A[n,i]*B[n,j]*C[n,k]
    return OUT

import numpy
#
def vec_tensor_product_bcast_1(A,B,C):
    return A[:,:,None,None]*B[:,None,:,None]*C[:,None,None,:]

def vec_tensor_product_bcast_2(A,B,C):
    return (A[:,:,None]*B[:,None,:])[:,:,:,None]*C[:,None,None,:]


### guvectorize version
def tensor_product_core(A,B,C,OUT):
    N_A = A.shape[0]
    N_B = B.shape[0]
    N_C = C.shape[0]
    for i in range(N_A):
        for j in range(N_B):
            for k in range(N_C):
                OUT[i,j,k] = A[i]*B[j]*C[k]

from numba import float64, guvectorize, void
sig = void(float64[:],float64[:],float64[:],float64[:,:,::])
csig = '(a),(b),(c) -> (a,b,c)'
# sig = float64[:,:,:](float64[:],float64[:],float64[:])

guvec_tensor_product = guvectorize([sig],csig,nopython=True,target='cpu')(tensor_product_core)


N = 10000
N_a = 10
N_b = 20
N_c = 20
A = numpy.random.random((N,N_a))
B = numpy.random.random((N,N_b))
C = numpy.random.random((N,N_c))
OUT = numpy.zeros((N,N_a,N_b,N_c))

# warmup
oo = vec_tensor_product(A,B,C)
oo = guvec_tensor_product(A,B,C)
oo1 = vec_tensor_product_bcast_1(A,B,C)
oo2 = vec_tensor_product_bcast_2(A,B,C)



with timeit("jit"):
    for i in range(10): oo = vec_tensor_product(A,B,C)
with timeit("guvectorize"):
    for i in range(10): oo1 = guvec_tensor_product(A,B,C)
with timeit("jit (different ordering)"):
    for i in range(10): oo2 = vec_tensor_product_2(A,B,C)
with timeit("broadcast"):
    for i in range(10):  oo3 = vec_tensor_product_bcast_1(A,B,C)
with timeit("broadcast (grouped)"):
    for i in range(10):  oo4 = vec_tensor_product_bcast_2(A,B,C)
with timeit("einsum"):
    for i in range(10):  oo5 = numpy.einsum('ni,nj,nk->nijk',A,B,C)

assert(abs(oo2-oo).max()<1e-16)
assert(abs(oo1-oo).max()<1e-16)
assert(abs(oo3-oo).max()<1e-16)
assert(abs(oo4-oo).max()<1e-16)
assert(abs(oo5-oo).max()<1e-16)

# assert(abs(oo2-oo).max()<1e-16)

# special tensor reduction:
# we want to implement   C.(X⨂Y⨂Z)
# without computing the tensor product first

import numpy
import numpy as np
from numba import jit, guvectorize, float64, void

@jit
def vec_strange_tensor(C,X,Y,Z):
    N = X.shape[0]
    n_x = X.shape[1]
    n_y = Y.shape[1]
    n_z = Z.shape[1]
    res = np.zeros(N)
    for n in range(N):
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    res[n] += C[i,j,k]*X[n,i]*Y[n,j]*Z[n,k]
    return res

# @jit
def strange_tensor(C,X,Y,Z,OUT):
    n_x,n_y,n_z = C.shape
    OUT[0] = 0
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                OUT[0] += C[i,j,k]*X[i]*Y[j]*Z[k]

gu_strange_tensor_cpu = guvectorize([void(float64[:,:,:],float64[:],float64[:],float64[:],float64[:])],'(a,b,c),(a),(b),(c)->()',target='cpu')(strange_tensor)
gu_strange_tensor_par = guvectorize([void(float64[:,:,:],float64[:],float64[:],float64[:],float64[:])],'(a,b,c),(a),(b),(c)->()',target='parallel')(strange_tensor)


N = 10000
I = 10
J = 10
K = 10
C = numpy.random.random((I,J,K))
X = numpy.random.random((N,I))
Y = numpy.random.random((N,J))
Z = numpy.random.random((N,K))


with timeit("guvec"):
    for k in range(K): res_0 = gu_strange_tensor_cpu(C,X,Y,Z)

with timeit("guvec (parallel)"):
    for k in range(K): res_1 = gu_strange_tensor_par(C,X,Y,Z)

with timeit("jit"):
    for k in range(K): res_2 = vec_strange_tensor(C,X,Y,Z)

with timeit("direct"):
    for k in range(K): res_3 = (vec_tensor_product_bcast_1(X,Y,Z)*C.reshape((I,J,K))[None,:,:,:]).sum(axis=(1,2,3))



assert(abs(res_1-res_0).max()<1e-8)
assert(abs(res_2-res_0).max()<1e-8)
assert(abs(res_3-res_0).max()<1e-8)


#######################
#######################

from dolo.numeric.tensor import mdot
from dolo.misc.timing import timeit

import numpy as np

K = 100
A = np.random.random((20,8,8,8))
C = np.random.random((8,7))


from numba import jit

@jit(nopython=True)
def jmdot(A,X,Y,Z):
    n_x_1, n_x_2 = X.shape
    n_y_1, n_y_2 = Y.shape
    n_z_1, n_z_2 = Z.shape
    n_a = A.shape[0]
    T = numpy.zeros((n_a, n_x_2, n_y_2, n_z_2))
    for n in range(n_a):
        for k_1 in range(n_x_1):
            for k_2 in range(n_y_1):
                for k_3 in range(n_z_1):
                    t = A[n,k_1,k_2,k_3]
                    for i_1 in range(n_x_2):
                        for i_2 in range(n_y_2):
                            for i_3 in range(n_z_2):
                                T[n,i_1,i_2,i_3] += t*X[k_1,i_1]*Y[k_2,i_2]*Z[k_3,i_3]
    return T


def emdot(A,X,Y,Z):
    return np.einsum('nijk,ix,jy,kz->nxyz',A,X,Y,Z)


with timeit("original"):
    for k in range(K): res_1 = mdot(A,[C,C,C])

with timeit("special"):
    for k in range(K): res_2 = jmdot(A,C,C,C)

with timeit("einsum"):
    for k in range(K): res_3 = emdot(A,C,C,C)


assert( abs(res_1 - res_2).max() < 1e-10 )
assert( abs(res_1 - res_3).max() < 1e-10 )
