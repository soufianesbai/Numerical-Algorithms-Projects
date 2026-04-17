def integration_n_left_rectangle(f, a, b, N):
    """
    Approximate the integral of a function using the left rectangle method.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    N : number of subintervals
    """
    h = (b - a) / N  # Step size
    # Sum the function values at the left endpoints
    result = sum(f(a + i * h) for i in range(N))
    return h * result  # Multiply by step size to get the integral approximation

def integration_n_right_rectangle(f, a, b, N):
    """
    Approximate the integral of a function using the right rectangle method.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    N : number of subintervals
    """
    h = (b - a) / N  # Step size
    # Sum the function values at the right endpoints
    result = sum(f(a + (i + 1) * h) for i in range(N))
    return h * result  # Multiply by step size to get the integral approximation

def integration_n_middle_point(f, a, b, N):
    """
    Approximate the integral of a function using the midpoint rectangle method.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    N : number of subintervals
    """
    h = (b - a) / N  # Step size
    # Sum the function values at the midpoints of the subintervals
    result = sum(f(a + (i + 1/2) * h) for i in range(N))
    return h * result  # Multiply by step size to get the integral approximation

def integration_n_trapezoidal(f, a, b, N):
    """
    Approximate the integral of a function using the trapezoidal method.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    N : number of subintervals
    """
    h = (b - a) / N  # Step size
    # Apply trapezoidal method: average of function values at endpoints + sum of inner points
    result = 0.5 * (f(a) + f(b)) + sum(f(a + i * h) for i in range(1, N))
    return h * result  # Multiply by step size to get the integral approximation

def integration_n_simpson(f, a, b, N):
    """
    Approximate the integral of a function using Simpson's method.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    N : number of subintervals (must be even)
    """
    if N % 2 != 0:  # Ensure N is even
        N += 1  
    h = (b - a) / N  # Step size
    result = f(a) + f(b)  # Start with function values at the endpoints
    # Sum the function evaluations with alternating coefficients (4 and 2)
    for i in range(1, N):
        coeff = 4 if i % 2 != 0 else 2
        result += coeff * f(a + i * h)
    return (h / 3) * result  # Multiply by h/3 to get the integral approximation

def integration_eps(f, a, b, eps, meth, N_max, plot=False):
    """
    Approximate the integral with a specified method, iterating until the desired error is reached.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    eps : tolerance for error
    meth : integration method function (e.g., left, right, midpoint, trapezoidal, or Simpson)
    N_max : maximum number of iterations
    plot : boolean flag for plotting (default is False)
    """
    N = 1
    I = meth(f, a, b, N)  # Initial integral approximation
    count = 0
    # Loop until desired accuracy is achieved or max iterations reached
    while N < N_max:
        N *= 2  # Double N for better accuracy
        I_new = meth(f, a, b, N)  # New integral approximation
        if abs(I_new - I) < eps:  # If error is within tolerance, break
            break
        I = I_new
        count += 1
    return (I, count) if plot else I  # Return result and iteration count if plot is True

def integration_eps_middle_point(f, a, b, eps, N_max, plot=False):
    """
    Approximate the integral using the midpoint method with adaptive refinement based on error tolerance.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    eps : tolerance for error
    N_max : maximum number of iterations
    plot : boolean flag for plotting (default is False)
    """
    N = 1
    I = integration_n_middle_point(f, a, b, N)  # Initial midpoint approximation
    count = 0
    while N < N_max:
        N *= 3  # Increase N in steps of 3 for faster convergence
        I_new = integration_n_middle_point(f, a, b, N)
        if abs(I_new - I) < eps:  # If error is within tolerance, break
            break
        I = I_new
        count += 1
    return (I, count) if plot else I  # Return result and iteration count if plot is True

def integration_eps_simpson(f, a, b, eps, N_max, plot=False):
    """
    Approximate the integral using Simpson's method with adaptive refinement based on error tolerance.
    Arguments:
    f : function to integrate
    a, b : integration bounds
    eps : tolerance for error
    N_max : maximum number of iterations
    plot : boolean flag for plotting (default is False)
    """
    N = 2  # Start with an even N for Simpson's method
    I = integration_n_simpson(f, a, b, N)  # Initial Simpson's approximation
    count = 0
    while N < N_max:
        N *= 2  # Double N for better accuracy
        I_new = integration_n_simpson(f, a, b, N)
        if abs(I_new - I) < eps:  # If error is within tolerance, break
            break
        I = I_new
        count += 1
    return (I, count) if plot else I  # Return result and iteration count if plot is True
