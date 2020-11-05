from .splines import CubicSpline, CubicSplines
from .multilinear import LinearSpline, LinearSplines
from .eval_splines import options, eval_linear, eval_cubic, eval_spline
from .prefilter_cubic import filter_cubic, prefilter
from .option_types import options as extrap_options


# dummy functions
def UCGrid(*args):
    """
    Convert a tuple ( args.

    Args:
    """
    return tuple(args)
def CGrid(*args):
    """
    Returns a tuple of ( args and kw )

    Args:
    """
    return tuple(args)
def nodes(grid):
    """
    Return a list of all nodes in a grid.

    Args:
        grid: (todo): write your description
    """
    from interpolation.cartesian import mlinspace
    return mlinspace([g[0] for g in grid],[g[1] for g in grid],[g[2] for g in grid])
