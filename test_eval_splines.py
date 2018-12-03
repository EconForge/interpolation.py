from eval_splines import *
from interpolation.multilinear.mlinterp import mlinterp


N = 1000
K = 100
d = 2
grid = (
    (0.0, 1.0, K),
)*d
grid_nu = (
    (0.0, 1.0, K),
    np.linspace(0,1,K)
)
n_x = 6

V = np.random.random((K,K))
VV = np.random.random((K,K,n_x))

C = np.random.random((K+2,K+2))
CC = np.random.random((K+2,K+2,n_x))

points = np.random.random((N,2))

out = np.random.random((N))
Cout = np.random.random((N,n_x))

eval_linear(grid, V, points, out)
# eval_cubic(grid, C, points[0,:], out[0,0])
eval_linear(grid, VV, points, Cout)
eval_linear(grid, VV, points[0,:], Cout[0,:])

print("OK 1")

mlinterp(grid, V,points)
eval_linear(grid, V, points)
eval_linear(grid, V, points)
eval_linear(grid, V, points, out)

print("OK 2")
eval_linear(grid, V, points[0,:])
res_0 = eval_linear(grid, VV, points)
res_1 = eval_linear(grid, VV, points[0,:])


#nonuniform grid:
res_0_bis = eval_linear(grid_nu, VV, points)
res_1_bis = eval_linear(grid_nu, VV, points[0,:])

assert abs(res_0-res_0_bis).max()<1e-10
assert abs(res_1-res_1_bis).max()<1e-10


print("OK 3")
eval_cubic(grid, C, points, out)
# eval_cubic(grid, C, points[0,:], out[0,0])
eval_cubic(grid, CC, points, Cout)
eval_cubic(grid, CC, points[0,:], Cout[0,:])

print("OK 4")
eval_cubic(grid, C, points)
eval_cubic(grid, C, points[0,:])
eval_cubic(grid, CC, points)
eval_cubic(grid, CC, points[0,:])

print("OK 5")
exit()

####
###
###


N=100000
K = 100
d = 3
a = np.array([0.0]*d)
b = np.array([1.0]*d)
n = np.array([K]*d)
grid = ((0.0,1.0,K),)*d

CC = np.random.random((K+2,K+2,K+2,5))
point = np.random.random(3)
points = np.random.random((N,3))
vval = np.zeros(5)
vval1 = np.zeros(5)
vals = np.zeros((N,5))
vals1 = np.zeros((N,5))
from interpolation.splines.eval_cubic import eval_cubic_splines_3, vec_eval_cubic_splines_3

eval_cubic_splines_3(a,b,n,CC,point,vval)
eval_cubic(grid, CC, point, vval1)
abs(vval-vval1).max()

vec_eval_cubic_splines_3(a,b,n,CC,points,vals)
eval_cubic(grid, CC, points, vals1)
dd = vals1-vals
assert (abs(dd[~np.isnan(dd)]).max())<1e-10





vvals2 = eval_cubic(grid,CC,point)
vals2 = eval_cubic(grid,CC,points)

dd2 = vals2-vals
assert (abs(dd2).max())<1e-10
