from interpolation.linear_bases.chebychev import ChebychevBasis
from interpolation.linear_bases.linear import UniformLinearSpline
from interpolation.linear_bases.product import TensorBase

import numpy as np

n = 10
n_1 = 10
n_2 = 20
cb = ChebychevBasis(min=0,max=1,n=n_1)
lb = UniformLinearSpline(min=0,max=1,n=n_2)

tp = TensorBase([cb, lb])

print( tp.B(tp.grid).shape )

xvec = np.linspace(0,1,10)
yvec = np.linspace(0,1,20)

def f(x,y):
    return x**2 + y**3/(1+y)

values = f(tp.grid[:,0], tp.grid[:,1]).reshape((n_1,n_2))

coeffs = tp.filter(values)
coeffs_2 = tp.filter(values, filter=False)

assert(abs(coeffs-coeffs_2).max()<1e-8)


# try to evaluate the interpolant

tvec = np.linspace(0,1,100)
s = np.concatenate([tvec[:,None],tvec[:,None]],axis=1)
Phi = tp.Phi(s)

vv = Phi*coeffs_2
true_vals = [f(v,v) for v in tvec]


# take derivative w.r.t. various coordinates
Phi_0 = tp.Phi(s,orders=[1,0]).as_matrix()
Phi_1 = tp.Phi(s,orders=[0,1]).as_matrix()
Phi_1.shape

# does not work yet: should compute all necessary derivatives
# Phi_diff = tp.Phi(s, orders=[[0,1],[0,1],[0,1]])


# multivalued functions

vvalues = np.concatenate([values[:,:,None],values[:,:,None]],axis=2)
ccoeffs = tp.filter(vvalues)
abs(ccoeffs[:,:,0] - coeffs).max()<1e-8
abs(ccoeffs[:,:,1] - coeffs).max()<1e-8

# assert( abs( vv - true_vals ).max() < 1e-6 )
from matplotlib import pyplot as plt
plt.plot(tvec, vv)
plt.plot(tvec, true_vals)
# plt.plot(tvec, true_vals-vv)
plt.show()


plt.imshow(coeffs)


plt.imshow(coeffs_2)
