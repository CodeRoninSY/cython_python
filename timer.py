#!/usr/bin/env python
# -*- codimg: utf-8 -*-
import time
import functools


def timeit(func):
    ''' timeit '''
    @functools.wraps(func)
    def wrapper_timeit(*args, **kwargs):
        strt = time.perf_counter()
        value = func(*args, **kwargs)
        endt = time.perf_counter()
        runt = endt - strt
        print(f"Finished {func.__name__!r} in {runt:.6f} sec.")
        return value
    return wrapper_timeit
