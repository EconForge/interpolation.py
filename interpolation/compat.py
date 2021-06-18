from distutils.version import LooseVersion

from numba import __version__

if LooseVersion(__version__) >= "0.43":
    overload_options = {"strict": False}
else:
    overload_options = {}

if LooseVersion(__version__) >= "0.49":
    from numba.core.types import Array, Float, Integer
    from numba.core.types import Tuple, UniTuple
else:
    from numba.types import Array, Float, Integer
    from numba.types.containers import Tuple, UniTuple
