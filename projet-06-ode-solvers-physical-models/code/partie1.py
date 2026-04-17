import numpy as np
import matplotlib.pyplot as plt

"""
Representation chosen for a Cauchy problem:
- a vector (array) for the initial solution y0
- a python function f(t, y) that returns an array with
the size of y for the differential function
"""


def step_euler(y, t, h, f):
    """
    Performs one step of the Euler method for solving ODEs.

    Params:
        y (array of float) : current value of the dependent variable
        t (float) : current time
        h (float) : step size
        f (function) : derivative function f(t, y)

    Returns:
        y_next (array of float) : estimated value after one Euler step
    """
    return y + h * f(t, y)


def step_midpoint(y, t, h, f):
    """
    Performs one step of the midpoint method for solving ODEs.

    Params:
        y (array of float) : current value of the dependent variable
        t (float) : current time
        h (float) : step size
        f (function) : derivative function f(t, y)

    Returns:
        y_next (array of float) : estimated value after one midpoint step
    """
    y_middle = y + h / 2 * f(t, y)
    return y + h * f(t + h / 2, y_middle)


def step_heun(y, t, h, f):
    """
    Performs one step of the Heun method for solving ODEs.

    Params:
        y (array of float) : current value of the dependent variable
        t (float) : current time
        h (float) : step size
        f (function) : derivative function f(t, y)

    Returns:
        y_next (array of float) : estimated value after one Heun step
    """
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + (h / 2) * (k1 + k2)


def step_rk4(y, t, h, f):
    """
    Performs one step of the Range Kutta method for solving ODEs.

    Params:
        y (array of float) : current value of the dependent variable
        t (float) : current time
        h (float) : step size
        f (function) : derivative function f(t, y)

    Returns:
        y_next (array of float) : estimated value after one Range Kutta step
    """
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def meth_n_step(y0, t0, N, h, f, step_method):
    """
    Solves an ODE using a given one-step integration method over N steps.

    Params:
        y0 (array of float) : initial value of the dependent variable
        t0 (float) : initial time
        N (int) : number of integration steps
        h (float) : step size
        f (function) : derivative function f(t, y)
        step_method (function) : one-step integration method

    Returns:
        ts (array) : array of time points
        ys (array) : array of solution values at each time point
    """
    ys = [y0]
    ts = [t0]
    y, t = y0, t0

    for _ in range(N):
        y = step_method(y, t, h, f)
        t += h
        ys.append(y)
        ts.append(t)

    return np.array(ts), np.array(ys)


def meth_epsilon(y0, t0, tf, eps, f, step_method):
    """
    Solves an ODE on [t0, tf] using adaptive step refinement until a desired accuracy is reached.

    Params:
        y0 (array of float) : initial value of the dependent variable
        t0 (float) : initial time
        tf (float): final time
        eps (float) : desired accuracy (tolerance)
        f (function) : derivative function f(t, y)
        step_method (function) : one-step integration method

    Returns:
        ys (array) : array of solution values at each time point (with step size h)
    """
    N = 10

    while True:
        h = (tf - t0) / N
        _, yN = meth_n_step(y0, t0, N, h, f, step_method)
        _, y2N = meth_n_step(y0, t0, 2 * N, h / 2, f, step_method)
        error = np.max(np.abs(yN - y2N[::2]))
        if error < eps:
            return y2N[::2]
        N *= 2


def plot_tangent_field(f_vec, xlim, ylim, density, scale):
    """
    Plots the tangent (vector) field of a 2D autonomous system defined by f_vec.

    Params:
        function f_vec(x, y) that returns a 2D vector [dx/dt, dy/dt] evaluated at (x, y)
        xlim (tuple of float) : (xmin, xmax) range for the x-axis
        ylim (tuple of float) : (ymin, ymax) range for the y-axis
        density (int) : number of grid points along each axis
        scale (float) : factor to scale the length of the arrows in the plot.

    Returns:
        None
    """
    xs = np.linspace(xlim[0], xlim[1], density)
    ys = np.linspace(ylim[0], ylim[1], density)
    X, Y = np.meshgrid(xs, ys)  # Create a grid of points
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    # Compute the tangent vectors at each point and normalize them
    for i in range(density):
        for j in range(density):
            v = f_vec(X[i, j], Y[i, j])
            norm = np.linalg.norm(v)
            if norm != 0:
                v = v / norm
            U[i, j], V[i, j] = v

    plt.quiver(X, Y, U, V, pivot="mid", scale=scale, color="0.3", width=0.003)
