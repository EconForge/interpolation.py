import numba

# from numba import jitclass


# Extrapolation types

# the following fails because of https://github.com/EconForge/interpolation.py/issues/52
# spec = [
# ]
# @jitclass(spec)
# class c_CONSTANT:
#     def __init__(self):
#         pass
# @jitclass(spec)
# class c_LINEAR:
#     def __init__(self):
#         pass
# @jitclass(spec)
# class c_NEAREST:
#     def __init__(self):
#         pass
# CONSTANT = c_CONSTANT()
# LINEAR = c_LINEAR()
# NEAREST = c_NEAREST()


# this is a horrible workaround
CONSTANT = ((None,), (None,) * 1)
LINEAR = ((None,), (None,) * 2)
NEAREST = ((None,), (None,) * 3)


t_CONSTANT = numba.typeof(CONSTANT)
t_LINEAR = numba.typeof(LINEAR)
t_NEAREST = numba.typeof(NEAREST)

extrap_types = (t_CONSTANT, t_LINEAR, t_NEAREST)


tt = numba.typeof(CONSTANT)


from collections import namedtuple

_extrap = namedtuple("extrapolation_options", ["CONSTANT", "LINEAR", "NEAREST"])
options = _extrap(CONSTANT, LINEAR, NEAREST)
