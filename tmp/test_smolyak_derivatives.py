from numpy import *
from interpolation.smolyak import *

d = 2
mu = 2

def f(s):
    return (s**2).sum(axis=1)

# def f(s):
#     return concatenate( [ ((s**i).sum(axis=1))[:,None] for i in [2,3]] )

sg = SmolyakGrid(d, mu)

values_on_grid = f(sg.grid)
print(values_on_grid.shape)

sg = SmolyakInterp(sg,values_on_grid)

from interpolation.cartesian import cartesian

N = 5
x, y = [linspace(-1,1,N),linspace(-1,1,N)]
prod = cartesian([x,y])
# print( prod)

vals, vals_S, vals_c, vals_X = sg.interpolate(prod, deriv=True, deriv_th=True, deriv_X=True)
# print(vals.shape)
# print(vals_S.shape)
# print(vals_c.shape)
# print(vals_X.shape)
vv = f(prod)
# print(vals)
# from matplotlib.pyplot import *

# print( sum(abs(vals_c)>1-8) )
# print( sum(abs(vals_X)>1-8) )
# print(array(vals_c.shape).prod())

# print("so far so good")
from new_class import Smolyak, MultiSmolyak

a,b = [-1,-1],[1,1]
sg = MultiSmolyak(a,b,mu, values=values_on_grid[:,None])

new_vals = sg.interpolate(prod)
new_vals = new_vals.ravel()


# print(new_vals)
# print(vals)
# print(abs(new_vals-vals).max())


nv, nv_s, trash  = sg.interpolate(prod, deriv=True)
# print(nv.shape)
# print(nv_s.shape)


nv, nv_s, nv_x  = sg.interpolate(prod, deriv=True, deriv_X=True)

theta_0 = sg.__sinterp__[0].theta
# print(nv.shape)
# print(nv_s.shape)
# print(nv_x.shape)

#Test
print("TEST")
i_point = 10
i_out = 8
epsilon = 0.001
vvalues_on_grid = values_on_grid.copy()[:,None]
print(vvalues_on_grid.shape)
vvalues_on_grid[i_point,0] += epsilon
sg2 = MultiSmolyak(a,b,mu, values=vvalues_on_grid)
new_vals_2 = sg2.interpolate(prod).ravel()

theta_1 = sg2.__sinterp__[0].theta

diff_theta = (theta_1-theta_0)/epsilon

from dprint import dprint

# print("Diff:", str(diff_theta))
# print(diff_theta[1])
# print(nv_s.shape)
# print(nv_s[1,...])

diff = (new_vals_2 - new_vals)/epsilon
dprint(diff.shape)

if True:
    dprint(diff[i_out])

print(nv_x[i_out,:,i_point])
#
# # imshow(vals.reshape((len(x),len(y))))
# figure()
# imshow(vv.reshape((len(x),len(y))))
#
#
#
# figure()
# spy(vals_c)
# title('coefficients')
# figure()
# spy(vals_X)
# title('coefficients')
#
# show()
