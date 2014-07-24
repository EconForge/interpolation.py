from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Splines',
  ext_modules = cythonize("eval_cubic_splines_cython.pyx"),
)
