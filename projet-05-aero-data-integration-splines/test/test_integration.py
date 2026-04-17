import matplotlib.pyplot as plt
import numpy as np
from math import *
import sys
sys.path.append('../src')
import integration

methods = [
    integration.integration_n_left_rectangle,
    integration.integration_n_right_rectangle,
    integration.integration_n_trapezoidal,
    integration.integration_n_middle_point,
    integration.integration_n_simpson
]

method_names = [
    "Left Rectangle",
    "Right Rectangle",
    "Trapezoidal",
    "Middle Point",
    "Simpson"
]

def test_polynomial():
    print("Polynomial")
    f = lambda x: 4 * x + 5
    for i, meth in enumerate(methods):
        integral = meth(f, 0, 10, 100)
        print(f"{method_names[i]:<15} → {integral:.10f}")
    print()

def test_cos():
    print("Cos")
    f = lambda x: cos(x)
    for i, meth in enumerate(methods):
        integral = meth(f, 0, 10, 100)
        print(f"{method_names[i]:<15} → {integral:.10f}")
    print()

def test_sin():
    print("Sin")
    f = lambda x: sin(x)
    for i, meth in enumerate(methods):
        integral = meth(f, 0, 10, 100)
        print(f"{method_names[i]:<15} → {integral:.10f}")
    print()

def test_tan():
    print("Tan")
    f = lambda x: tan(x)
    for i, meth in enumerate(methods):
        try:
            integral = meth(f, 0, 10, 100)
            print(f"{method_names[i]:<15} → {integral:.10f}")
        except Exception as e:
            print(f"{method_names[i]:<15} → Error: {e}")
    print()

def test_double_step():
    print("Double step")
    f = lambda x: sin(x)
    for i in range(3):  
        integral, count = integration.integration_eps(f, 0, 10, 1e-9, methods[i], 10000, plot=True)
        print(f"{method_names[i]:<15} → {integral:.10f} (subdivisions: {2**count})")
    
    integral, count = integration.integration_eps_middle_point(f, 0, 10, 1e-9, 10000, plot=True)
    print(f"{'Middle Point (eps)':<20} → {integral:.10f} (subdivisions: {3**count})")

    integral, count = integration.integration_eps_simpson(f, 0, 10, 1e-9, 10000, plot=True)
    print(f"{'Simpson (eps)':<20} → {integral:.10f} (subdivisions: {2**(count+1)})")
    print()

def convergence_plot():
    f = lambda x : sin(x)
    a, b = 0, 10
    N_values = [2**i for i in range(2, 12)]
    exact = -cos(10) + cos(0)

    errors = {
        "Left Rectangle": [],
        "Right Rectangle": [],
        "Trapezoidal": [],
        "Middle Point": [],
        "Simpson": []
    }

    for N in N_values:
        errors["Left Rectangle"].append(abs(integration.integration_n_left_rectangle(f, a, b, N) - exact))
        errors["Right Rectangle"].append(abs(integration.integration_n_right_rectangle(f, a, b, N) - exact))
        errors["Trapezoidal"].append(abs(integration.integration_n_trapezoidal(f, a, b, N) - exact))
        errors["Middle Point"].append(abs(integration.integration_n_middle_point(f, a, b, N) - exact))
        errors["Simpson"].append(abs(integration.integration_n_simpson(f, a, b, N) - exact))

    for method, err in errors.items():
        plt.plot(N_values, err, label=method)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("N (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("Convergence speed of integration methods")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
if __name__ == "__main__":
    test_polynomial()
    test_cos()
    test_sin()
    test_tan()
    test_double_step()
    convergence_plot()