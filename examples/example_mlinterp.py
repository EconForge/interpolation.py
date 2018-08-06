import numpy as np

##########
# interp #
##########

### 1d interpolation

from interpolation import interp

x = np.linspace(0,1,100)**2 # non-uniform points
y = np.linspace(0,1,100)    # values

# interpolate at one point:
interp(x,y,0.5)

# or at many points:
u = np.linspace(0,1,1000)   # points
interp(x,y,u)

# one can iterate at low cost since the function is jitable:
from numba import njit
@njit
def vec_eval(u):
    N = u.shape[0]
    out = np.zeros(N)
    for n in range(N):
        out[n] = interp(x,y,u)
    return out

print( abs(vec_eval(u) - interp(x,y,u)).max())


### 2d interpolation (same for higher orders)

from interpolation import interp

x1 = np.linspace(0,1,100)**2 # non-uniform points
x2 = np.linspace(0,1,100)**2 # non-uniform points
y = np.array([[np.sqrt(u1**2 + u2**2) for u2 in x2] for u1 in x1])
# (y[i,j] = sqrt(x1[i]**2+x2[j]**2)


# interpolate at one point:
interp(x1,x2,y,0.5,0.2)
interp(x1,x2,y,(0.5,0.2))

# or at many points: (each line corresponds to one observation)
points = np.random.random((1000,2))
interp(x1,x2,y,points)

from numba import njit
@njit
def vec_eval(p):
    N = u.shape[0]
    out = np.zeros(N)
    for n in range(N):
        z1 = p[n,0]
        z2 = p[n,1]
        out[n] = interp(x1,x2,y,z1,z2)
    return out

print( abs(vec_eval(points) - interp(x1,x2,y,points)).max())


# in the special case where the points at which one wants to interpolate
# form a cartesian grid, one can use another call style:

z1 = np.linspace(0,1,100)
z2 = np.linspace(0,1,100)
out = interp(x1,x2,y,z1,z2)
# out[i,j] contains f(z1[i],z2[j])




############
# mlinterp #
############

# same as interp but with less flexible and more general API

from interpolation import mlinterp

x1 = np.linspace(0,1,100)**2 # non-uniform points for first dimensoin
x2 = (0,1,100) # uniform points for second dimension
grid = (x1,x2)
y = np.array([[np.sqrt(u1**2 + u2**2) for u2 in x2] for u1 in x1])


points = np.random.random((1000,2))

# vectorized call:
mlinterp(grid, y, points)

# non-vectorized call (note third argument must be a tuple of floats of right size)
mlinterp(grid, y, (0.4, 0.2))
