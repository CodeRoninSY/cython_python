#!/bin/bash
# <2019-11-18> CodeRoninSY
# Automate compile and build extension library (ode_py.so)
# GNU gcc >> ode_py.cpython-36m-x86_64-linux-gnu.so
python setup_ode_py.py build_ext --inplace
python ode_py.py