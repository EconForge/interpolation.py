import numpy as np
 
a = np.array([3.0,2.0,4.0])
b = np.array([5.0,5.0,8.0])
n = np.array([23,4,4])
c = np.random.random((25,6,6))
cc = np.random.random((25,6,6,7))
p = np.random.random(3)
pp = np.random.random((10000,3))

from interpolation.splines.eval_splines import eval_cubic_spline
from interpolation.splines.eval_cubic import eval_cubic_spline as eval_old


res0 = eval_cubic_spline(a,b,n,c,p)
res1 = eval_old(a,b,n,c,p)

print(res0)
print(res1)



from interpolation.splines.eval_splines import vec_eval_cubic_spline
from interpolation.splines.eval_cubic import vec_eval_cubic_spline as vec_eval_old


res0 = vec_eval_cubic_spline(a,b,n,c,pp)
res1 = vec_eval_old(a,b,n,c,pp)

print(abs(res0 - res1).max())

import time
K = 10000
t1 = time.time()
for k in range(K):
    res0 = vec_eval_cubic_spline(a,b,n,c,pp)
t2 = time.time()
for k in range(K):
    res1 = vec_eval_old(a,b,n,c,pp)
t3 = time.time()

print("NEW: {}".format(t2-t1))
print("OlD: {}".format(t3-t2))

from interpolation.splines.eval_splines import eval_cubic_splines
from interpolation.splines.eval_cubic import eval_cubic_splines as eval_old


res0 = eval_cubic_splines(a,b,n,cc,p)
res1 = eval_old(a,b,n,cc,p)

print(res0)
print(res1)

from interpolation.splines.eval_splines import vec_eval_cubic_splines
from interpolation.splines.eval_cubic import vec_eval_cubic_splines as eval_old


res0 = vec_eval_cubic_splines(a,b,n,cc,pp)
res1 = eval_old(a,b,n,cc,pp)

print(abs(res0 - res1).max())

import time
K = 10000
t1 = time.time()
for k in range(K):
    res0 = vec_eval_cubic_splines(a,b,n,cc,pp)
t2 = time.time()
for k in range(K):
    res1 = eval_old(a,b,n,cc,pp)
t3 = time.time()


print("NEW: {}".format(t2-t1))
print("OLD: {}".format(t3-t2))