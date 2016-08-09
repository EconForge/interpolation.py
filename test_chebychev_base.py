from matplotlib import pyplot as plt
from interpolation.linear_bases.chebychev import ChebychevBasis
import numpy as np


uf = ChebychevBasis(-0.5, 1.5, 10)
x = np.linspace(-0.7,1.7, 100)

uf.eval(0.1)
uf.eval([0.1,0.2])


mat = uf.eval(x)
for i in range(10):
    plt.plot(x, mat[:,i])
plt.xlim(-0.55, 1.55)
plt.ylim(-1.5, 1.5)


def fun(x): return np.sin(x**2)


vals = fun(uf.nodes)
c = uf.filter(vals)

# evaluate approximated function
Phi = uf(x)
interp_values = Phi@c

# evaluate approximated function
dPhi = uf(x,orders=1)
d_interp_values = dPhi@c

plt.figure()
plt.plot(x,fun(x))
plt.plot(x,interp_values)
plt.plot(uf.nodes,fun(uf.nodes),'o')
plt.plot(x,d_interp_values)
