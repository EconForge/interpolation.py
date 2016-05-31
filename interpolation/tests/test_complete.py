import numpy as np
import pandas as pd

from interpolation.complete_poly import (_complete_poly_impl,
                                         _complete_poly_impl_vec,
                                         complete_polynomial,
                                         n_complete)


def prof_complete_poly_impl(ns=[2, 3, 4, 5, 6], ds=[1, 2, 3, 4, 5],
                            Ts=[100, 250, 1000, 10000, 50000,
                                100000, 500000, 1000000],
                            nrep=10):
    pass
