import numpy as np

from interpolation.linear_bases.spline import CubicSplineBasis, UniformCubicSplineBasis

sp = CubicSplineBasis(np.linspace(0,1,5))
sp_2 = UniformCubicSplineBasis(0,1,5)




sp.n
sp.k

len(sp.nodes)
len(sp.knots)

x = np.linspace(-0.1, 1.1, 100)

B = sp.eval(x,orders=[0,0])

from matplotlib import pyplot as plt

for i in range(B.shape[1]):
    plt.plot(x, B[:,i])
for n in sp.nodes:
    plt.vlines(n, -2,2,linestyle='--')
for n in sp.knots:
    plt.vlines(n, -2,2,linestyle=':')

plt.ylim(-2,2)
plt.show()

x = np.random.random(10)

c = sp.filter(x)
