import numba

from numba import jitclass

spec = [
]

# Extrapolation types

@jitclass(spec)
class c_CONSTANT:
    def __init__(self):
        pass


@jitclass(spec)
class c_LINEAR:
    def __init__(self):
        pass

@jitclass(spec)
class c_NEAREST:
    def __init__(self):
        pass

CONSTANT = c_CONSTANT()
LINEAR = c_LINEAR()
NEAREST = c_NEAREST()

t_CONSTANT = numba.typeof(CONSTANT)
t_LINEAR = numba.typeof(LINEAR)
t_NEAREST = numba.typeof(NEAREST)

extrap_types = (t_CONSTANT, t_LINEAR, t_NEAREST)


tt = numba.typeof(CONSTANT)


from collections import namedtuple

_extrap = namedtuple("extrapolation_options", ["CONSTANT", 'LINEAR', 'NEAREST'])
options = _extrap(CONSTANT, LINEAR, NEAREST)


