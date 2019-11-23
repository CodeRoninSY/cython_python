#!/usr/bin/env python
"""
# ode_py.pyx
#
# <2019-11-18> CodeRoninSY
#
# Synopsis: Cythonize version of "ode_py.py"
#
# Compile cython pyx file using "setup.py" build file
# $> python setup_ode_py.py build_ext --inplace
#
"""
from timer import timeit
import numpy as np
import math
cimport cython
from cython.parallel import prange
from pprint import pprint

cdef class Function:
    cpdef double evaluate(self, double y, double t) except *:
        return 0

f = Function()


def feval(funcName, *args):
    return eval(funcName)(*args)


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def euler(f, double y0, double [:] t):
    ''' Euler explicit method '''
    cdef int n, i
    cdef double [:] y

    n = len(t)
    y = np.array( [y0] * n)
    for i in range( n - 1 ):
        y[i+1] = y[i] + ( t[i+1] - t[i]) * f(y[i], t[i])

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def heun(f, double y0, double [:] t):
    ''' Heun 2nd order explicit '''
    cdef int n, i
    cdef double [:] y

    n = len( t )
    y = np.array( [y0] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( y[i], t[i] )
        k2 = h * f( y[i] + k1, t[i+1] )
        y[i+1] = y[i] + ( k1 + k2 ) / 2.0

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def rk2a(f, double y0, double [:] t):
    ''' Runge-Kutta 2nd order $O(h^2)$'''
    cdef int n, i
    cdef double [:] y
    cdef double h
    cdef double k1

    n = len(t)
    y = np.array([y0]*n)
    for i in range( n-1 ):
        h = t[i+1] - t[i]
        k1 = h * f(y[i], t[i]) / 2.0
        y[i+1] = y[i] + h * f(y[i]+k1, t[i]+h/2.0)

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def rk2b(f, double y0, double [:] t):
    ''' Runge-Kutta 2nd order $O(h^2)$'''
    cdef int n, i
    cdef double [:] y
    cdef double h
    cdef double k1, k2

    n = len(t)
    y = np.array([y0]*n)
    for i in range( n-1 ):
        h = t[i+1] - t[i]
        k1 = h * f(y[i], t[i])
        k2 = h * f(y[i] + k1, t[i+1])
        y[i+1] = y[i] + (k1 + k2) /2.0

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def rk4(f, double y0, double [:] t):
    ''' Runge-Kutta 4th order $O(h^4)$'''
    cdef int n, i
    cdef double [:] y
    cdef double h
    cdef double k1, k2, k3, k4

    n = len(t)
    y = np.array([y0]*n)

    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f( y[i], t[i] )
        k2 = h * f( y[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( y[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( y[i] + k3, t[i+1] )
        y[i+1] = y[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def rk45(f, double y0, double [:] t):
    ''' Runge-Kutta 4th(5) order $O(h^4(5))$'''
    cdef int n, i
    cdef double [:] y, e
    cdef double k1, k2, k3, k4, k5, k6, y5

    cdef double c20, c30, c40, c50, c60
    cdef double c21, c31, c41, c42, c43, c51, c52, c53, c54
    cdef double c61, c62, c63, c64, c65
    cdef double a1, a2, a3, a4, a5
    cdef double b1, b2, b3, b4, b5, b6

    # Coefficients used to compute the independent variable argument of f
    c20  =   2.500000000000000e-01  #  1/4
    c30  =   3.750000000000000e-01  #  3/8
    c40  =   9.230769230769231e-01  #  12/13
    c50  =   1.000000000000000e+00  #  1
    c60  =   5.000000000000000e-01  #  1/2

    # Coefficients used to compute the dependent variable argument of f
    c21 =   2.500000000000000e-01  #  1/4
    c31 =   9.375000000000000e-02  #  3/32
    c32 =   2.812500000000000e-01  #  9/32
    c41 =   8.793809740555303e-01  #  1932/2197
    c42 =  -3.277196176604461e+00  # -7200/2197
    c43 =   3.320892125625853e+00  #  7296/2197
    c51 =   2.032407407407407e+00  #  439/216
    c52 =  -8.000000000000000e+00  # -8
    c53 =   7.173489278752436e+00  #  3680/513
    c54 =  -2.058966861598441e-01  # -845/4104
    c61 =  -2.962962962962963e-01  # -8/27
    c62 =   2.000000000000000e+00  #  2
    c63 =  -1.381676413255361e+00  # -3544/2565
    c64 =   4.529727095516569e-01  #  1859/4104
    c65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate
    a1  =   1.157407407407407e-01  #  25/216
    a2  =   0.000000000000000e-00  #  0
    a3  =   5.489278752436647e-01  #  1408/2565
    a4  =   5.353313840155945e-01  #  2197/4104
    a5  =  -2.000000000000000e-01  # -1/5

    b1  =   1.185185185185185e-01  #  16.0/135.0
    b2  =   0.000000000000000e-00  #  0
    b3  =   5.189863547758284e-01  #  6656.0/12825.0
    b4  =   5.061314903420167e-01  #  28561.0/56430.0
    b5  =  -1.800000000000000e-01  # -9.0/50.0
    b6  =   3.636363636363636e-02  #  2.0/55.0

    n = len( t )
    y = np.array( [ y0 ] * n )
    e = np.array( [ 0 * y0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( y[i], t[i] )
        k2 = h * f( y[i] + c21 * k1, t[i] + c20 * h )
        k3 = h * f( y[i] + c31 * k1 + c32 * k2, t[i] + c30 * h )
        k4 = h * f( y[i] + c41 * k1 + c42 * k2 + c43 * k3, t[i] + c40 * h )
        k5 = h * f( y[i] + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4, \
                        t[i] + h )
        k6 = h * f( \
            y[i] + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5, \
            t[i] + c60 * h )

        y[i+1] = y[i] + a1 * k1 + a3 * k3 + a4 * k4 + a5 * k5
        y5 = y[i] + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6

        e[i+1] = abs( y5 - y[i+1] )

    return ( y, e )


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def rkf(f, double a, double b, double y0, double tol, \
        double hmax, double hmin):
    ''' Runge-Kutta 6th order $O(h^6)$'''
    cdef int n, i
    cdef double t, y, h
    cdef double [:] Y, T
    cdef double a2, a3, a4, a5, a6
    cdef double b21, b31, b32, b41, b51, b52, b53, b54, \
                b61, b62, b63, b64, b65
    cdef double r1, r2, r3, r4, r5, r6
    cdef double c1, c2, c3, c4, c5

    # Coefficients used to compute the independent variable argument of f
    a2  =   2.500000000000000e-01  #  1/4
    a3  =   3.750000000000000e-01  #  3/8
    a4  =   9.230769230769231e-01  #  12/13
    a5  =   1.000000000000000e+00  #  1
    a6  =   5.000000000000000e-01  #  1/2

    # Coefficients used to compute the dependent variable argument of f
    b21 =   2.500000000000000e-01  #  1/4
    b31 =   9.375000000000000e-02  #  3/32
    b32 =   2.812500000000000e-01  #  9/32
    b41 =   8.793809740555303e-01  #  1932/2197
    b42 =  -3.277196176604461e+00  # -7200/2197
    b43 =   3.320892125625853e+00  #  7296/2197
    b51 =   2.032407407407407e+00  #  439/216
    b52 =  -8.000000000000000e+00  # -8
    b53 =   7.173489278752436e+00  #  3680/513
    b54 =  -2.058966861598441e-01  # -845/4104
    b61 =  -2.962962962962963e-01  # -8/27
    b62 =   2.000000000000000e+00  #  2
    b63 =  -1.381676413255361e+00  # -3544/2565
    b64 =   4.529727095516569e-01  #  1859/4104
    b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.
    r1  =   2.777777777777778e-03  #  1/360
    r3  =  -2.994152046783626e-02  # -128/4275
    r4  =  -2.919989367357789e-02  # -2197/75240
    r5  =   2.000000000000000e-02  #  1/50
    r6  =   3.636363636363636e-02  #  2/55

    # Coefficients used to compute 4th order RK estimate
    c1  =   1.157407407407407e-01  #  25/216
    c3  =   5.489278752436647e-01  #  1408/2565
    c4  =   5.353313840155945e-01  #  2197/4104
    c5  =  -2.000000000000000e-01  # -1/5

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
    t = a
    y = y0
    h = hmax

    # Initialize arrays that will be returned
    T = np.array( [t] )
    Y = np.array( [y] )

    while t < b:
        # Adjust step size when we get to last interval
        if t + h > b:
            h = b - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.
        k1 = h * f( y, t )
        k2 = h * f( y + b21 * k1, t + a2 * h )
        k3 = h * f( y + b31 * k1 + b32 * k2, t + a3 * h )
        k4 = h * f( y + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h )
        k5 = h * f( y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h )
        k6 = h * f( y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                    t + a6 * h )

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
        r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        if len( np.shape( r ) ) > 0:
            r = max( r )
        if r <= tol:
            t = t + h
            y = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T = np.append( T, t )
            Y = np.append( Y, [y], 0 )

        # Now compute next step size, and make sure that it is not too big or
        # too small.
        h = h * min( max( 0.84 * ( tol / r )**0.25, 0.1 ), 4.0 )

        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error: stepsize should be smaller than %e." % hmin)
            break

    # endwhile
    return ( T, Y )


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def dopri5(f, double a, double b, double y0, double tol, double hmax, double hmin, int maxiter):
    ''' Runge-Kutta Dormand-Prince 5th order $O(h^5)$'''
    cdef int n, i, flag
    cdef double t, y, h
    cdef double [:] Y, T
    cdef double a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, \
                a61, a62, a63, a64, a65, \
                a71, a72, a73, a74, a75, a76
    cdef double c2, c3, c4, c5, c6, c7
    cdef double b1, b2, b3, b4, b5, b6, b7
    cdef double b1p, b2p, b3p, b4p, b5p, b6p, b7p
    cdef double K1, K2, K3, K4, K5, K6, K7
    cdef double error, delta

    # we trust that the compiler is smart enough to pre-evaluate the
    # value of the constants.
    a21 = (1.0/5.0)
    a31 = (3.0/40.0)
    a32 = (9.0/40.0)
    a41 = (44.0/45.0)
    a42 = (-56.0/15.0)
    a43 = (32.0/9.0)
    a51 = (19372.0/6561.0)
    a52 = (-25360.0/2187.0)
    a53 = (64448.0/6561.0)
    a54 = (-212.0/729.0)
    a61 = (9017.0/3168.0)
    a62 = (-355.0/33.0)
    a63 = (46732.0/5247.0)
    a64 = (49.0/176.0)
    a65 = (-5103.0/18656.0)
    a71 = (35.0/384.0)
    a72 = (0.0)
    a73 = (500.0/1113.0)
    a74 = (125.0/192.0)
    a75 = (-2187.0/6784.0)
    a76 = (11.0/84.0)

    c2 = (1.0 / 5.0)
    c3 = (3.0 / 10.0)
    c4 = (4.0 / 5.0)
    c5 = (8.0 / 9.0)
    c6 = (1.0)
    c7 = (1.0)

    b1 = (35.0/384.0)
    b2 = (0.0)
    b3 = (500.0/1113.0)
    b4 = (125.0/192.0)
    b5 = (-2187.0/6784.0)
    b6 = (11.0/84.0)
    b7 = (0.0)

    b1p = (5179.0/57600.0)
    b2p = (0.0)
    b3p = (7571.0/16695.0)
    b4p = (393.0/640.0)
    b5p = (-92097.0/339200.0)
    b6p = (187.0/2100.0)
    b7p = (1.0/40.0)

    t = a
    y = y0
    h = hmax

    # Initialize arrays that will be returned
    T = np.array( [t] )
    Y = np.array( [y] )

    for i in range(maxiter):
        # /* Compute the function values */
        # print(i, Y[i], T[i])
        K1 = f(y, t)
        K2 = f(y + h*(a21*K1), t + c2*h)
        K3 = f(y+h*(a31*K1+a32*K2), t + c3*h)
        K4 = f(y+h*(a41*K1+a42*K2+a43*K3), t + c4*h )
        K5 = f(y+h*(a51*K1+a52*K2+a53*K3+a54*K4), t + c5*h)
        K6 = f(y+h*(a61*K1+a62*K2+a63*K3+a64*K4+a65*K5), t + h)
        K7 = f(y+h*(a71*K1+a72*K2+a73*K3+a74*K4+a75*K5+a76*K6), t + h)

        error = abs((b1-b1p)*K1+(b3-b3p)*K3+(b4-b4p)*K4+(b5-b5p)*K5 +
                    (b6-b6p)*K6+(b7-b7p)*K7)

        # error control
        delta = 0.84 * math.pow(tol / error, (1.0/5.0))
        if (error < tol):
            t = t + h
            y = y + h * (b1*K1+b3*K3+b4*K4+b5*K5+b6*K6)
            T = np.append(T, t)
            Y = np.append( Y, [y], 0)

        if (delta <= 0.1):
            h = h * 0.1
        elif (delta >= 4.0):
            h = h * 4.0
        else:
            h = delta * h

        if (h > hmax):
            h = hmax

        if (t >= b):
            flag = 0
            break
        elif (t + h > b):
            h = b - t
        elif (h < hmin):
            flag = 1
            break
    maxiter = maxiter-i
    if (i <= 0):
        flag = 2

    return (T, Y, flag,  maxiter)



@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def pc4( f, double y0, double [:] t ):
    ''' Adams-Bashforth 4th order predictor-corrector $O(h^4)$'''
    cdef int n, i
    cdef double [:] y
    cdef double f0, f1, f2, f3, w, fw
    cdef double h, k1, k2, k3, k4

    n = len( t )
    y = np.array( [ y0 ] * n )

    # Start up with 4th order Runge-Kutta (single-step method).
    # The extra code involving f0, f1, f2, and f3 helps us get ..
    # ready for the multi-step method to follow in order to ..
    #  minimize the number of function evaluations needed.

    f1 = f2 = f3 = 0
    for i in range( min( 3, n - 1 ) ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        k1 = h * f0
        k2 = h * f( y[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( y[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( y[i] + k3, t[i+1] )
        y[i+1] = y[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        f1, f2, f3 = ( f0, f1, f2 )

    # Begin Adams-Bashforth-Moulton steps

    for i in range( 3, n - 1 ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        w = y[i] + h * ( 55.0 * f0 - 59.0 * f1 + 37.0 * f2 - 9.0 * f3 ) / 24.0
        fw = f( w, t[i+1] )
        y[i+1] = y[i] + h * ( 9.0 * fw + 19.0 * f0 - 5.0 * f1 + f2 ) / 24.0
        f1, f2, f3 = ( f0, f1, f2 )

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def ab2e( f, double y0, double [:] t ):
    ''' Adams-Bashforth 2nd order explicit predictor-corrector $O(h^2)$'''
    cdef int n, i
    cdef double [:] y
    cdef double f0, f1
    cdef double h

    n = len( t )
    y = np.array( [ y0 ] * n )

    # Start up with 1st order Euler (single-step method).
    # The extra code involving f0, f1 helps us get ..
    # ready for the multi-step method to follow in order to ..
    #  minimize the number of function evaluations needed.

    f1 = 0
    for i in range( min( 1, n - 1 ) ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        y[i+1] = y[i] + h * f0
        f1 = f0

    # Begin Adams-Bashforth-Moulton steps

    for i in range( 1, n - 1 ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        y[i+1] = y[i] + h * ( 3.0 * f0 - f1 ) / 2.0
        f1 = f0

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def am2i( f, double y0, double [:] t ):
    ''' Adams-Moulton 2nd order implicit predictor-corrector $O(h^2)$'''
    cdef int n, i
    cdef double [:] y
    cdef double f0, f1, f2
    cdef double h

    n = len( t )
    y = np.array( [ y0 ] * n )

    # Start up with 1st order Euler (single-step method).
    # The extra code involving f0, f1 helps us get ..
    # ready for the multi-step method to follow in order to ..
    #  minimize the number of function evaluations needed.

    f1 = f2 = 0
    for i in range( min( 2, n - 1 ) ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        k1 = h * f0
        k2 = h * f( y[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( y[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( y[i] + k3, t[i+1] )
        y[i+1] = y[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        f1, f2 = (f0, f1)

    # Begin Adams-Bashforth-Moulton steps

    for i in range( 2, n - 1 ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        w = y[i] + h * f0
        fw = f( w, t[i+1] )
        y[i+1] = y[i] + h * ( 5.0 * fw + 8.0 * f0 - f1 ) / 12.0
        f1, f2 = ( f0, f1 )

    return y


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def ms4( f, double y0, double [:] t ):
    ''' Milne-Simpson 4th order predictor-corrector $O(h^4)$
    Do not use this method - needs to be fixed...
    '''
    cdef int n, i
    cdef double [:] y
    cdef double f0, f1, f2, f3, f4
    cdef double h, k1, k2, k3, k4

    n = len( t )
    y = np.array( [ y0 ] * n )

    # Start up with 4th order Runge-Kutta (single-step method).
    # The extra code involving f0, f1, f2, and f3 helps us get ..
    # ready for the multi-step method to follow in order to ..
    #  minimize the number of function evaluations needed.

    f1 = f2 = f3 = 0
    for i in range( min( 3, n - 1 ) ):
        h = t[i+1] - t[i]
        f0 = f( y[i], t[i] )
        k1 = h * f0
        k2 = h * f( y[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( y[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( y[i] + k3, t[i+1] )
        y[i+1] = y[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    f1 = f(y[1], t[1])
    f2 = f(y[2], t[2])
    f3 = f(y[3], t[3])

    # Begin Adams-Bashforth-Moulton steps

    for i in range( 3, n - 1 ):
        h = t[i+1] - t[i]
        p = y[i-2] + (4.0/3.0) * h * ( 2.0 * f1 - f2 + 2.0 * f3);

        f4 = f( p, t[i] + h )
        y[i+1] = y[i-1] + h * ( f2 + 4.0 * f3 + f4 ) / 3.0
        f1, f2, f3 = ( f2, f3, f(y[i+1], t[i+1]) )

    return y
