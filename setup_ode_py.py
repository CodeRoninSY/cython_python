#!/usr/bin/env python
'''
# <2019-11-18> CodeRoninSY
# compile: $> python setup_ode_py.py build_ext --inplace
'''
from distutils.core import setup
from Cython.Build import cythonize

setup(name='ODE_py',
      ext_modules=cythonize("ode_py.pyx"))
