import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sys
sys.path.append('../src')
import spline

import load_foil

# ================== Visual test on x -> x ** 3 ==================
x = np.linspace(0, 2, 1000)           # Control points (x)
y = x ** 3                         # Control points (y = x³)

# Compute second derivatives for the cubic spline
M = spline.cubic_spline(x, y)

# Evaluate spline 
x_eval = np.linspace(0, 2, 100)
y_spline = np.array([spline.eval_spline(x, y, M, xi) for xi in x_eval])

# Plot the points and the spline interpolation
#plt.plot(x, y, 'ro', label="Control points")
plt.plot(x_eval, y_spline, 'b-', label="Cubic spline of $x^3$")
plt.plot(x_eval, x_eval**3, 'g--', label="Function $x^3$")
plt.legend()
plt.title("Spline interpolation of x → x³")
plt.grid()
plt.show()

# ================== Numerical test on x -> x ** 3 ==================
y_exact = x_eval ** 3
relative_error = norm(y_spline - y_exact) / norm(y_exact)
assert np.isclose(relative_error, 0.0, atol=1e-10), "Spline interpolation error is not close to zero"

# ================== Visual test on x -> sin(x) ==================
x = np.linspace(0, 2 * np.pi, 1000)           # Control points (x)
y = np.sin(x)                         # Control points (y = sin(x))

# Compute second derivatives for the cubic spline
M = spline.cubic_spline(x, y)

# Evaluate spline
x_eval = np.linspace(0, 2 * np.pi, 100)
y_spline = np.array([spline.eval_spline(x, y, M, xi) for xi in x_eval])

# Plot the points and the spline interpolation
plt.plot(x_eval, y_spline, 'b-', label="Cubic spline of $\\sin(x)$")
plt.plot(x_eval, np.sin(x_eval), 'g--', label="Function $\\sin(x)$")
plt.legend()
plt.title("Spline interpolation of x → sin(x)")
plt.grid()
plt.show()

# ================== Numerical test on x -> sin(x) ==================
y_exact= np.sin(x_eval)
y_spline = np.array([spline.eval_spline(x, y, M, xi) for xi in x_eval])
relative_error = norm(y_spline - y_exact) / norm(y_exact)
assert np.isclose(relative_error, 0.0, atol=1e-10), "Spline interpolation error is not close to zero"

# ================== Visual test on airfoil ==================
dim, ex, ey, ix, iy = load_foil.load_foil("m5.dat")

M_ex = spline.cubic_spline(ex, ey)
x_eval = np.linspace(min(ex), max(ex), 500)
y_spline_ex = [spline.eval_spline(ex, ey, M_ex, xi) for xi in x_eval]

M_ix = spline.cubic_spline(ix, iy)
y_spline_ix = [spline.eval_spline(ix, iy, M_ix, xi) for xi in x_eval]

plt.plot(ex, ey, 'ro', label="Upper part")  
plt.plot(x_eval, y_spline_ex, 'b-', label="Cubique spline of upper part")  

plt.plot(ix, iy, 'ro', label="Lower part") 
plt.plot(x_eval, y_spline_ix, 'b-', label="Cubique spline of lower part")  

plt.axis("equal")  
plt.legend()
plt.title("")
plt.grid(True)

plt.show()

print("All spline tests passed successfully!")