from interpolation.complete_poly import complete_inds, n_complete, complete_polynomial, _complete_poly_impl


[i for i in complete_inds(2,3)]


n_complete(4,2)


import numpy
points = numpy.array([
    [0.1, 0.2],
    [0.1, 0.3],
    [0.3, 0.5],
    [0.4, 0.5]
])

Phi = complete_polynomial(points,3)
Phi_2 = complete_polynomial((points-1)/2,3)


Phi_sim = complete_polynomial(s_sim.T, deg).T

# evaluate future controls at each future state
_complete_poly_impl(S.T, deg, Phi_sim.T)
np.dot(Phi_sim, coefs, out=X)

# updated coefficients
_complete_poly_impl(s_sim.T, deg, Phi_sim.T)
new_coefs = np.ascontiguousarray(lstsq(Phi_sim, new_x)[0])


_complete_poly_impl(s_sim.T, deg, Phi_sim.T)
new_coefs = np.ascontiguousarray(lstsq(Phi_sim, new_x)[0])

from scipy.linalg import lstsq
import numpy as np

class CompletePolynomial(object):

    def __init__(self, n, d):
        self.n = n
        self.d = d

    def fit_values(self, s, x):
        Phi = complete_polynomial(s.T, self.d).T
        self.Phi = Phi
        self.coefs = np.ascontiguousarray(lstsq(Phi, x)[0])

    def __call__(self, s):

        Phi = complete_polynomial(s.T, self.d).T
        return np.dot(Phi, self.coefs)



def f(x,y): return x
def f2(x,y): return x**2 + y**2

from interpolation.cartesian import cartesian


points = numpy.random.random((1000,2))
vals = f(points[:,0], points[:,1])

cp = CompletePolynomial(2, 3)
cp.fit_values(points, vals)




cp(points) - vals

cp.Phi.shape



vals2 = np.column_stack( [vals, f2(points[:, 0], points[:, 1])] )
cp.fit_values(points, vals2)
cp(points) - vals2



inds = [i for i in complete_inds(2,3)]


for h in zip(inds, cp.coefs):
    print(h)

nmax = 2
for ntot in range(nmax+1):
    for i_0 in range(1+ntot):
        i_1 = ntot-i_0
        print((i_0, ntot-i_0))


nmax = 2
for ntot in range(nmax+1):
    for i_0 in range(1+ntot):
        for i_1 in range(1+ntot-i_0):
            i_2 = ntot-i_0-i_1
            print((i_0, i_1, i_2))

def get_I_max(imax):
    s = '1+ntot'
    if imax>0:
        s += '-' + str.join('-',['i_{}'.format(i) for i in range(imax)])
    return s

def get_complete_product_code(d, target='generator'):

    if target=='generator':
        s = 'def complete_product(nmax):'
        indent = 1
    else:
        indent = 0
    s += '''
{}for ntot in range(nmax+1):
'''.format('    '*indent)
    if d>1:
        for k in range(d-1):
            indent += 1
            line = 'for i_{} in range({}):\n'.format(k, get_I_max(k))
            s += '    '*(indent) + line
        indent += 1
        s += '    '*(indent) + 'i_{} = ntot-'.format(d-1) + str.join('-', ['i_{}'.format(i) for i in range(d-1)]) + '\n'
    else:
        s += '    '*(indent-1) + 'i_0 = ntot\n'

    # do smthg
    if target=='generator':
        s += '    '*(indent) + 'yield ({})'.format(str.join(',',  ['i_{}'.format(i)  for i in range(d)]))
    elif target=='print':
        s += '    '*(indent) + 'print( ({}) )'.format(str.join(',',  ['i_{}'.format(i)  for i in range(d)]))

    return s

s = ( get_complete_product_code(3) )
print(s)
import ast
expr = ast.parse(s)
exec(s)

for i in complete_product(0):

    print(i)

def my_complete_poly(points, n):
    N = points.shape[0]
    d = points.shape[1]
    powers = numpy.zeros((N,d,n+1))
    powers[:,:,0] = 1
    for k in range(n):
        powers[:,:,k+1] = powers[:,:,k]*points

    Phi = numpy.ones((N, n_complete(n,d)))
    for l,inds in enumerate(complete_product(n)):
        for i,k in enumerate(inds):
            Phi[:,l] *= powers[:,i,k]

    return Phi

Phi.shape

Phi
N = 10000
d = 3
points = numpy.random.random((N, d))

%time phiphi = complete_polynomial(points.T, 5).T
phiphi.shape

%time Phi = my_complete_poly(points, 5)

a1 = np.array( sorted( Phi.ravel() ) )
a2 = np.array( sorted( phiphi.ravel() ) )

abs( a1 - a2 ).max()
