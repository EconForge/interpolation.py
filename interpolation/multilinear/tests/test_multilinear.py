from numpy import linspace, array
from numpy.random import random
from numba import typeof

import numpy as np
from interpolation.multilinear.fungen import get_index


def test_barycentric_indexes():

    # irregular grid
    gg = np.array([0.0, 1.0])
    assert get_index(gg, -0.1) == (0, -0.1)
    assert get_index(gg, 0.5) == (0, 0.5)
    assert get_index(gg, 1.1) == (0, 1.1)

    # regular grid
    gg = (0.0, 1.0, 2)
    assert get_index(gg, -0.1) == (0, -0.1)
    assert get_index(gg, 0.5) == (0, 0.5)
    assert get_index(gg, 1.1) == (0, 1.1)


# 2d-vecev-scalar
a2 = (linspace(0, 1, 10), random((10)), random((200, 1)))
# 2d-pointev-scalar
a3 = (linspace(0, 1, 10), random((10)), array([0.5]))
# 2d-tupev-scalar
a4 = (linspace(0, 1, 10), random((10)), (0.5,))
# 2d-fev-scalar
a5 = (linspace(0, 1, 10), random((10)), 0.5)

# 2d-carev-vec
b1 = (linspace(0, 1, 10), random((10, 3)), linspace(0, 1, 200))
# 2d-vecev-vec
b2 = (linspace(0, 1, 10), random((10, 3)), random((200, 1)))
# 2d-pointev-vec
b3 = (linspace(0, 1, 10), random((10, 3)), array([0.5]))
# 2d-tupev-vec
b4 = (linspace(0, 1, 10), random((10, 3)), (0.5,))  # unsupported
# 2d-fev-vec
b5 = (linspace(0, 1, 10), random((10, 3)), 9.5)  # unsupported


# 2d-carev-scalar
c1 = (
    linspace(0, 1, 10),
    linspace(0, 1, 20),
    random((10, 20)),
    linspace(0, 1, 200),
    linspace(0, 1, 200),
)
# 2d-vecev-scalar
c2 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20)), random((200, 2)))
# 2d-pointev-scalar
c3 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20)), array([0.5, 2.0]))
# 2d-tupev-scalar
c4 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20)), (0.5, 2.0))
# 2d-fev-scalar
c5 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20)), 0.5, 2.0)

# 2d-carev-vecvec
d1 = (
    linspace(0, 1, 10),
    linspace(0, 1, 20),
    random((10, 20, 3)),
    linspace(0, 1, 200),
    linspace(0, 1, 200),
)
# 2d-vecev-vec
d2 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20, 3)), random((200, 2)))
# 2d-pointev-vec
d3 = (linspace(0, 1, 10), linspace(0, 1, 20), random((10, 20, 3)), array([0.5, 2.0]))
# 2d-tupev-vec
d4 = (
    linspace(0, 1, 10),
    linspace(0, 1, 20),
    random((10, 20, 3)),
    (0.5, 2.0),
)  # unsupported (return type not known)
d5 = (
    linspace(0, 1, 10),
    linspace(0, 1, 20),
    random((10, 20, 3)),
    0.5,
    2.0,
)  # unsupported (return type not known)


tests = [a2, a3, a4, c1, c2, c3, c4]
tests_failing = [b1, b2, b3, b4, b5, d4, d5, d1, d2, d3, d4, d5]

from interpolation.multilinear.mlinterp import mlinterp, interp


def test_mlinterp():

    # simple multilinear interpolation api

    import numpy as np
    from interpolation import mlinterp

    # from interpolation.multilinear.mlinterp import mlininterp, mlininterp_vec
    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 20)
    y = np.random.random((10, 20))

    z1 = np.linspace(0, 1, 30)
    z2 = np.linspace(0, 1, 30)

    pp = np.random.random((2000, 2))

    res0 = mlinterp((x1, x2), y, pp)
    res0 = mlinterp((x1, x2), y, (0.1, 0.2))


def test_multilinear():

    # flat flexible api

    for t in tests:

        tt = [typeof(e) for e in t]
        print(tt)
        rr = interp(*t)

        try:
            print(f"{tt}: {rr.shape}")
        except:
            print(f"{tt}: OK")


#
# exit()
#
# ###############################################################
# # Now let's see what are the gains of jit for repeated callas #
# # with some unscientific performance benchmarks               #
# ###############################################################
#
# N = 100000
# points = np.random.rand(N,2)
#
#
#
#
# grid = (
#     (0.0, 1.0, 11),
#     (0.0, 1.0, 11)
# )
#
# vv = np.linspace(0,1,11)
#
# grid_uneven = (
#     vv,
#     vv
# )
#
# C = np.random.rand(11,11)
#
# # two equivalent calls:
# v = interp(grid, C, (0.3, 0.2))
# v_unevn = interp(grid_uneven, C, (0.3, 0.2))
#
# assert(abs(v_unevn-v)<1e-10)
#
#
# #
# # # let's compare with interp2d
# from scipy.interpolate import interp2d
# intp2 = interp2d(vv,vv,C.T)
# v_2d = intp2(0.3,0.2)
# assert(abs(v_2d-v)<1e-10)
#
# # and Regular Grid Interpolator
# from scipy.interpolate import RegularGridInterpolator
# vg = np.linspace(0,1,11)
# rgi = RegularGridInterpolator((vg,vg),C)
# v_rgi = rgi([0.3, 0.2])[0]
# assert(abs(v_rgi-v)<1e-10)
#
#
#
# vals = vec_interp(grid, C, points)
# vals_un = vec_interp(grid_uneven, C, points)
# vals_rgi = rgi(points)
#
# # both give the same result
# assert((abs(vals-vals_rgi).max()<1e-10))
# assert((abs(vals-vals_un).max()<1e-10))
# #
# # import time
# # K = 1000
# #
# # t1_a = time.time()
# # for k in range(K):
# #     vals = vec_interp(grid, C, points)
# # t1_b = time.time()
# #
# # t2_a = time.time()
# # for k in range(K):
# #     vals_un = vec_interp(grid_uneven, C, points)
# # t2_b = time.time()
# #
# # t3_a = time.time()
# # for k in range(K):
# #     vals_rgi = rgi(points)
# # t3_b = time.time()
# #
# # print(f"Even: {t1_b-t1_a}")
# # print(f"Uneven: {t2_b-t2_a}")
# # print(f"Scipy: {t3_b-t3_a}")
