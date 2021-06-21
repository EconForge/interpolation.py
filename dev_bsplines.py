from numba import jit


@jit(nopython=True)
def B0(u, i, x):

    if (u[i] <= x < u[i + 1]):
        return 1.0
    else:
        return 0.0

@jit(nopython=True)
def B(p, u, i, x):

    if (p == 0):
        return B0(u, i, x)
    else:
        return (((x - u[i]) / (u[i + p] - u[i])) * B(p - 1, u, i, x) + ((u[i + p + 1] - x) / (u[i + p + 1] - u[i + 1])) * B(p - 1, u, i + 1, x))



import numpy as np

m = 1000
u = np.linspace(0,1,m)
ufull = np.concatenate( [[u[0]-2*(u[1]-u[0])], [u[0]-(u[1]-u[0])], u, [u[-1]+(u[-1]-u[-2])], [u[-1]+2*(u[-1]-u[-2])]])
print(ufull)



@jit(nopython=True)
def construct_band():
    vals = []
    for i0 in range(2,m+2):
        u0 = ufull[i0]
        line = []
        for i in (i0-3, i0-2, i0-1):
            line.append( ( B(3, ufull, i, u0) ) )
        vals.append(line)

construct_band()

import time
t1 = time.time()

construct_band()

t2 = time.time()
print(f"Elapsed: {t2-t1}")