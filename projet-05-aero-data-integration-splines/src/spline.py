import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(x, y):
    """
    Implement the cubic spline interpolation algorithm.
    Arguments:
    x : array, the x values of the data points
    y : array, the corresponding y values of the data points
    Returns:
    M : array, the second derivatives at the spline points
    """
    n = len(x)
    h = np.diff(x)  # Step sizes between consecutive x values
    
    # Initialize the tridiagonal matrix A and the right-hand side vector B
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Construct the matrix A and vector B based on the spline equations
    for i in range(1, n-1):
        A[i, i-1] = h[i-1] / 6
        A[i, i] = (h[i-1] + h[i]) / 3
        A[i, i+1] = h[i] / 6
        B[i] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
    
    # Boundary conditions: set second derivatives at the ends to 0
    A[0, 0] = 1
    A[-1, -1] = 1
    B[0] = 0
    B[-1] = 0
    
    # Solve the linear system to get the second derivatives (M)
    V = np.linalg.solve(A, B)
    
    return V  # Return the second derivatives at the spline points

def eval_spline(x, y, M, xi):
    """
    Evaluate the cubic spline at the specified points.
    Arguments:
    x : array, the x values of the data points
    y : array, the corresponding y values of the data points
    M : array, the second derivatives at the spline points
    xi : array, the x values at which to evaluate the spline
    Returns:
    spline_values : the interpolated y values at xi
    """
    # Find the interval where xi falls
    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(x) - 2)  # Ensure the index is within bounds
    h = x[i+1] - x[i]  # Width of the interval
    
    # Coefficients for the cubic spline interpolation
    A = (x[i+1] - xi) / h
    B = (xi - x[i]) / h
    C = (1/6) * (A**3 - A) * h**2
    D = (1/6) * (B**3 - B) * h**2
    
    # Compute the interpolated value using the spline formula
    return A * y[i] + B * y[i+1] + C * M[i] + D * M[i+1]
