from packaging.version import parse

from numba import __version__

if parse(__version__) >= parse("0.43"):
    overload_options = {"strict": False}
else:
    overload_options = {}

if parse(__version__) >= parse("0.49"):
    from numba.core.types import Array, Float, Integer
    from numba.core.types import Tuple, UniTuple
else:
    from numba.types import Array, Float, Integer
    from numba.types.containers import Tuple, UniTuple
