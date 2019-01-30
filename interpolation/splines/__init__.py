from .splines import CubicSpline, CubicSplines
from .multilinear import LinearSpline, LinearSplines
from .eval_splines import options, eval_linear, eval_cubic
from .prefilter_cubic import filter_cubic
from .option_types import options as extrap_options

# dummy functions
def UCGrid(*args):
    return tuple(args)
def CGrid(*args):
    return tuple(args)
def nodes(grid):
    from interpolation.cartesian import mlinspace
    return mlinspace([g[0] for g in grid],[g[1] for g in grid],[g[2] for g in grid])
