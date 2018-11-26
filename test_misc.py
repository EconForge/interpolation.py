import numpy as np
from interpolation.multilinear.mlinterp import detect_types

d = 2
n_x= 3
a = (0.0, 0.0)
b = (1.0, 1.0)
n = (10,10)
c = np.random.random((12,12,n_x))
cc = c[..., 0]
N = 100
points = np.random.random((N,2))
out = np.empty((N,n_x))

import numba
tt = numba.typeof((a,b,n,c,out))

detect_types(tt)
