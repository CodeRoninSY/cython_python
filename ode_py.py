#!/usr/bin/env python
"""
ode_py.py

<2019-11-17> CodeRoninSY

* Synopsis: Several numerical ODE solver is implemented.
Euler, Heun, RK2a, RK2b, Runge-Kutta 4th order,
Runge-Kutta 5(4) order, Runge-Kutta-Fehlberg, RK Dormand-Prince 5(4),
Adams-Bashforth (4) predictor-corrector.

Python Scipy.integrate library is also worked out for learning.

* 1st order ODEs
y' = f(y,t) with y(t[0]) = y0

* Compile & run:
1. python setup_ode_py.py build_ext --inplace
2. python ode_py.py

* Required files:
# ode_py.pyx        : Cythonized python (developed)
# setup_ode_py.py   : Build setup
# ode_py.sh         : Build & run script (handles all compilation & execution)
# ode_py.py         : Main python driver script

* References:
1. Numerical Recipes, 3rd edition, W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
2. "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode"
3. "https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method"
4. "http://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html"
5. "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html"
6. "https://web.archive.org/web/20150907215914/http://adorio-research.org/wordpress/?p=6565"
7. "http://www.unige.ch/~hairer/prog/nonstiff/dopri5.f"
8. "http://people.bu.edu/andasari/courses/numericalpython/python.html"

"""

from __future__ import print_function
from timeit import timeit
import numpy as np
from pylab import *
from scipy.integrate import solve_ivp, ode
from ode_py import Function, euler, heun, rk2a, rk2b, rk4, \
                    rk45, rkf, pc4, dopri5


if __name__ == "__main__":
    # Setting parameters for scipy.ode methods
    WTHJACO = False     # with_jacobian parameter
    MAXSTP = 1000  # max iter no for scipy.integrate.ode solvers
    ADMSORDR1 = 4  # Adams method order; i=4
    ADMSORDR2 = 12  # Adams method order, i=12
    BDFORDR1 = 5  # BDF method order; <= 5
    BDFORDR2 = 2  # BDF method order; <= 5
    DOPRI5_SF = 0.84  # Dopri5 S: safety factor; default=0.9
    ATOL = 1.0e-6  # atol (dopri5)
    RTOL = 1.0e-6   # rtol (dopri5)
    ERTOL = 1.0e-6  # err tolerance
    BETA = 0.04     # controller test, PI

    def f(x, t) -> Function:
        return x * np.sin(t)

    def f1(t, y):
        return y * np.sin(t)


    # t_init, t_last
    a, b = (0.0, 10.0)
    # x_init:x(0) / y_init:y(0)
    x0 = -1.0

    t_init = [a, b]

    # discretize the solution domain
    n = 101
    t = np.linspace(a, b, n)

    # print(f"linspace-t: {t}")

    # compute various numerical solutions
    x_euler = euler(f, x0, t)
    x_heun = heun(f, x0, t)
    x_rk2a = rk2a(f, x0, t)
    x_rk2b = rk2b(f, x0, t)
    x_rk4 = rk4(f, x0, t)
    x_rk45, e_rk45 = rk45(f, x0, t)
    x_pc4 = pc4(f, x0, t)
    t_rkf, x_rkf = rkf(f, a, b, x0, ERTOL, 1.0, 0.01)  # unequally spaced t
    t_dp54, x_dp54, flag, maxiter = dopri5(
        f, a, b, x0, ERTOL, 1.0, 0.01, 1000 )

    print(f"Flag(dopri5): {flag}; Maxiter: {maxiter:8d}")

    # scipy.integrate.solve_ivp solutions
    t_evPt = np.array(t)
    sol1 = solve_ivp(f1, [0.0, 10.0], [-1], method='RK45',t_eval=t_evPt)
    sol2 = solve_ivp(f1, [0.0, 10.0], [-1], method='RK23', t_eval=t_evPt)
    sol3 = solve_ivp(f1, [0.0, 10.0], [-1], method='Radau', t_eval=t_evPt)
    sol4 = solve_ivp(f1, [0.0, 10.0], [-1], method='BDF', t_eval=t_evPt)
    sol5 = solve_ivp(f1, [0.0, 10.0], [-1], method='LSODA', t_eval=t_evPt)

    print(sol1.success, sol1.nfev, sol1.njev, sol1.status)
    print(sol2.success, sol2.nfev, sol2.njev, sol2.status)
    print(sol3.success, sol3.nfev, sol3.njev, sol3.status)
    print(sol4.success, sol4.nfev, sol4.njev, sol4.status)
    print(sol5.success, sol5.nfev, sol5.njev, sol5.status)

    # scipy.integrate.ode solutions
    dt = float((b - a) / (n - 1))
    t_l = b

    # Vode
    print(f"VODE, meth=ADAMS, ord: {ADMSORDR1}")
    v1_t = []
    v1_y = []
    v1 = ode(f1).set_integrator('vode', method='adams', \
        with_jacobian=WTHJACO, nsteps=MAXSTP, order=ADMSORDR1)
    v1.set_initial_value(x0, a)
    v1_t = np.append(v1_t, a)
    v1_y = np.append(v1_y, x0)
    while v1.successful() and v1.t < t_l:
        v1_t = np.append(v1_t, v1.t+dt)
        v1_y = np.append(v1_y, v1.integrate(v1.t+dt, step=False))
        print(v1.t, v1.y)

    # print(f"v1_t: {v1_t}, v1_y: {v1_y}")

    # scipy.integrate.ode solutions - VODE Adams(12)
    print(f"VODE, meth=ADAMS, ord: {ADMSORDR2}")
    v2_t = []
    v2_y = []
    v2 = ode(f1).set_integrator('vode', method='adams',
                                with_jacobian=WTHJACO, nsteps=MAXSTP, order=ADMSORDR2)
    v2.set_initial_value(x0, a)
    v2_t = np.append(v2_t, a)
    v2_y = np.append(v2_y, x0)
    while v2.successful() and v2.t < t_l:
        v2_t = np.append(v2_t, v2.t+dt)
        v2_y = np.append(v2_y, v2.integrate(v2.t+dt, step=False))
        print(v2.t, v2.y)


    # scipy.integrate.ode solutions - VODE BDF(5)
    print(f"VODE, meth=BDF, ord: {BDFORDR1}")
    v3_t = []
    v3_y = []
    v3 = ode(f1).set_integrator('vode', method='BDF',
                                with_jacobian=WTHJACO, nsteps=MAXSTP, order=BDFORDR1)
    v3.set_initial_value(x0, a)
    v3_t = np.append(v3_t, a)
    v3_y = np.append(v3_y, x0)
    while v3.successful() and v3.t < t_l:
        v3_t = np.append(v3_t, v3.t+dt)
        v3_y = np.append(v3_y, v3.integrate(v3.t+dt, step=False))
        print(v3.t, v3.y)

    # scipy.integrate.ode solutions - VODE BDF(2)
    print(f"VODE, meth=BDF, ord: {BDFORDR2}")
    v4_t = []
    v4_y = []
    v4 = ode(f1).set_integrator('vode', method='BDF',
                                with_jacobian=WTHJACO, nsteps=MAXSTP, order=BDFORDR2)
    v4.set_initial_value(x0, a)
    v4_t = np.append(v4_t, a)
    v4_y = np.append(v4_y, x0)
    while v4.successful() and v4.t < t_l:
        v4_t = np.append(v4_t, v4.t + dt)
        v4_y = np.append(v4_y, v4.integrate(v4.t+dt, step=False))
        print(v4.t, v4.y)

    # scipy.integrate.ode solutions - LSODA; BDF(5) & ADMS(12)
    print(f"LSODA, meth=BDF, ord: {BDFORDR2}")
    v5_t = []
    v5_y = []
    v5 = ode(f1).set_integrator('lsoda', method='BDF',
                                with_jacobian=WTHJACO, nsteps=MAXSTP, max_order_s=BDFORDR2)
    v5.set_initial_value(x0, a)
    v5_t = np.append(v5_t, a)
    v5_y = np.append(v5_y, x0)
    while v5.successful() and v5.t < t_l:
        v5_t = np.append(v5_t, v5.t + dt)
        v5_y = np.append(v5_y, v5.integrate(v5.t+dt, step=False))
        print(v5.t, v5.y)

    # scipy.integrate.ode solutions - DOPRI;
    print(f"DOPRI5, meth=RK, ord: 5")
    v6_t = []
    v6_y = []
    v6 = ode(f1).set_integrator('dopri5',
                                nsteps=MAXSTP, safety=DOPRI5_SF, atol=ATOL, rtol=RTOL,
                                beta=BETA)
    v6.set_initial_value(x0, a)
    v6_t = np.append(v6_t, a)
    v6_y = np.append(v6_y, x0)
    while v6.successful() and v6.t < t_l:
        v6_t = np.append(v6_t, v6.t + dt)
        v6_y = np.append(v6_y, v6.integrate(v6.t+dt, step=False))
        print(v6.t, v6.y)

    # scipy.integrate.ode solutions - DOP853;
    print(f"DOP853, meth=RK, ord: 8(5,3)")
    v7_t = []
    v7_y = []
    v7 = ode(f1).set_integrator('dop853',
                                nsteps=MAXSTP, safety=DOPRI5_SF, atol=ATOL, rtol=RTOL,
                                beta=BETA)
    v7.set_initial_value(x0, a)
    v7_t = np.append(v7_t, a)
    v7_y = np.append(v7_y, x0)
    while v7.successful() and v7.t < t_l:
        v7_t = np.append(v7_t, v7.t + dt)
        v7_y = np.append(v7_y, v7.integrate(v7.t+dt, step=False))
        print(v7.t, v7.y)

    #  compute true solution values in equal spaced and unequally spaced cases
    x = -np.exp(1.0 - np.cos(t))
    xrkf = -np.exp(1.0 - np.cos(t_rkf))
    xdp54 = -np.exp(1.0 - np.cos(t_dp54))
    y_ex = -np.exp(1.0 - np.cos(v1_t))     # analytical solution for scipy.ODE

    #  figure( 1 )
    figure(2, figsize=(14,10))
    subplot(2, 2, 1)
    plot(t, x_euler, 'b-.', t, x_heun, 'g-.', \
        t, x_rk2a, 'r-.', t, x_rk2b, 'y-.', t, x_pc4, 'c-.')
    xlabel('$t$')
    ylabel('$x$')
    title('Solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('Euler', 'Heun', '$O(h^2)$ Runge-Kutta', \
        'RK2B', 'PC4'), loc='lower left', \
           framealpha=0.2, fontsize='small')

    # figure( 2 )
    subplot(2, 2, 2)
    plot(t, x_euler - x, 'b-.', \
        t, x_heun - x, 'g-.', \
        t, x_rk2a - x, 'r-.', t, x_rk2b - x, 'y-.',t, x_pc4 - x, 'c-.')
    xlabel('$t$')
    ylabel('$x - x^*$')
    title('Errors in solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('Euler', 'Heun', '$O(h^2)$ Runge-Kutta', \
        'RK2B', 'PC4'), loc='upper left', \
           framealpha=0.2, fontsize='small')

    # figure( 3 )
    subplot(2, 2, 3)
    plot(t, x_rk4, 'b-.', \
        t_rkf, x_rkf, 'r-.', t_dp54, x_dp54, 'c-.', \
            t, x_rk45, 'g-.')
    xlabel('$t$')
    ylabel('$x$')
    title('Solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$O(h^4)$ Runge-Kutta',
            'Runge-Kutta-Fehlberg', 'Dormand-Prince', \
                'RK45'), loc='lower left', \
           framealpha=0.2, fontsize='small')

    # figure( 4 )
    subplot(2, 2, 4)
    plot(t, x_rk4 - x, 'b-.',  \
        t_rkf, x_rkf - xrkf, 'r-.', \
        t_dp54, x_dp54 - xdp54, 'c-.', t, x_rk45 - x, 'g-.')
    xlabel('$t$')
    ylabel('$x - x^*$')
    title('Errors in solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$O(h^4)$ Runge-Kutta',
            'Runge-Kutta-Fehlberg', 'Dormand-Prince', \
                'RK45'), loc='upper left', \
           framealpha=0.2, fontsize='small')

    # show()
    savefig('ode_py_Fig1.png')

    # flatten the scipy.integrate.solve_ivp y results for plotting
    y_1 = np.ravel(sol1.y)
    y_2 = np.ravel(sol2.y)
    y_3 = np.ravel(sol3.y)
    y_4 = np.ravel(sol4.y)
    y_5 = np.ravel(sol5.y)

    # figure( 1 )
    figure(3, figsize=(14, 10))
    subplot(2, 2, 1)
    plot(sol1.t, y_1, 'b-.', sol2.t, y_2, 'r-.', sol3.t, y_3, 'g-.', \
        sol4.t, y_4, 'c-.', sol5.t, y_5, 'm-.')
    xlabel('$t$')
    ylabel('$x$')
    title('Solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$S1$ RK45', '$S2$ RK23', '$S3$ Radau', \
        '$S4$ BDF', '$S5$ LSODA'), loc='lower left', \
           framealpha=0.2, fontsize='small')

    # figure( 2 )
    subplot(2, 2, 2)
    plot(sol1.t, y_1 - x, 'b-.', sol2.t, y_2 - x, 'r-.', \
        sol3.t, y_3 - x, 'g-.', sol4.t, y_4 - x, 'c-.', \
            sol5.t, y_5 - x, 'm-.' )
    xlabel('$t$')
    ylabel('$y - y^*$')
    title('Errors in solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$S1$ RK45', '$S2$ RK23', '$S3$ Radau',
            '$S4$ BDF', '$S5$ LSODA'), loc='upper left', \
           framealpha=0.2, fontsize='small')

    # figure( 3 )
    subplot(2, 2, 3)
    plot(v1_t, v1_y, 'b-.', v2_t, v2_y, 'r-.', v3_t, v3_y, 'g-.', \
        v4_t, v4_y, 'c-.', v5_t, v5_y, 'm-.', v6_t, v6_y, 'k-.', \
         v7_t, v7_y, 'y-.')
    xlabel('$t$')
    ylabel('$x$')
    title('Solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$Vode$ Adams(4)', '$Vode$ Adams(12)',
            '$Vode$ BDF(5)', '$Vode$ BDF(2)', \
            '$LSODA$ BDF=2', '$RKDP$ Dopri5', '$RKDP$ Dop853'), loc='lower left',
           framealpha=0.2, fontsize='small')

    # figure( 4 )
    subplot(2, 2, 4)
    plot(v1_t, v1_y - y_ex, 'b-.', v2_t, v2_y - y_ex, 'r-.', \
        v3_t, v3_y - y_ex, 'g-.', v4_t, v4_y - y_ex, 'c-.', \
        v5_t, v5_y - y_ex, 'm-.', v6_t, v6_y - y_ex, 'k-.', \
         v7_t, v7_y - y_ex, 'y-.')
    xlabel('$t$')
    ylabel('$x - x^*$')
    title('Errors in solutions of $dx/dt = x \sin t$, $x(0)=-1$')
    legend(('$Vode$ Adams(4)', '$Vode$ Adams(12)', \
            '$Vode$ BDF(5)', '$Vode$ BDF(2)', \
            '$LSODA$ BDF=2', '$RKDP$ Dopri5', '$RKDP$ Dop853'),
            loc='upper left', \
                framealpha=0.2, fontsize='small')


    # show()
    savefig('ode_py_Fig2.png')
