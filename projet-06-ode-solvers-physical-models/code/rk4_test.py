import numpy as np
import matplotlib.pyplot as plt
from partie1 import step_rk4, meth_epsilon, meth_n_step, plot_tangent_field
import pytest


def test_step_rk4_constant():
    """dy/dt = 1, y(0) = 0 ⇒ y(t) = t"""
    f = lambda t, y: 1
    sol = meth_n_step(y0=0, t0=0, N=10, h=0.1, f=f, step_method=step_rk4)
    # sol[i] ≈ i*0.1
    assert np.allclose(sol, np.linspace(0, 1, 11), atol=1e-6)


def test_meth_epsilon_accuracy():
    """Tests that meth_epsilon achieves accuracy on a single ODE."""
    f = lambda t, y: y  # y' = y, y(0) = 1 ⇒ y = e^t
    sol = meth_epsilon(y0=1, t0=0, tf=1.0, eps=1e-5, f=f, step_method=step_rk4)
    tvals = np.linspace(0, 1, len(sol))
    exact = np.exp(tvals)
    # last value to eps
    assert abs(sol[-1] - exact[-1]) < 1e-5


# -------------------------
#  Plots requested
# -------------------------
if __name__ == "__main__":
    # --- Tests ---
    test_step_rk4_constant()
    test_meth_epsilon_accuracy()

    # --- Scalar ODE y' = y/(1+t 2), y(0) = 1 ---
    t0, tf, y0 = 0.0, np.pi, 1.0
    eps = 1e-4
    sol = meth_epsilon(
        y0, t0, tf, eps, f=lambda t, y: y / (1 + t * t), step_method=step_rk4
    )
    N = len(sol) - 1
    tvals = np.linspace(t0, tf, N + 1)
    exact = np.exp(np.arctan(tvals))

    plt.figure(figsize=(6, 4))
    plt.plot(tvals, sol, label="Runge Kutta raffinée")
    plt.plot(tvals, exact, "--", label="Exprimée exacte")
    plt.title("y' = y/(1+t^2), y(0)=1")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()

    # --- Vector ODE rotation ---
    # Tangent field : used in quiver
    def rot_field(x, y):
        return np.array([-y, x])

    # Differential system for integration
    def rot_dyn(t, y):
        y1, y2 = y
        return np.array([-y2, y1])

    # Plot tangent field
    plt.figure(figsize=(6, 6))
    plot_tangent_field(rot_field, xlim=[-2, 2], ylim=[-2, 2], density=25, scale=50)

    # Integration with unique initial condition
    t0, tf = 0.0, 2 * np.pi
    N = 500
    h = (tf - t0) / N
    y0 = np.array([1.0, 0.0])  # given initial condition

    ts, ys = meth_n_step(y0, t0, N, h, rot_dyn, step_rk4)

    # Plot the unique path
    plt.plot(ys[:, 0], ys[:, 1], label="Solution avec $y(0) = [1, 0]$", color="red")

    plt.title("Champ des tangentes et solution unique de $y' = [-y_2, y_1]$")
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.axis("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # Plots y1(t) and y2(t) and exact components
    plt.figure(figsize=(8, 4))
    plt.plot(ts, ys[:, 0], label="y1 numérique (Runge Kutta)")
    plt.plot(ts, ys[:, 1], label="y2 numérique (Runge Kutta)")
    plt.plot(ts, np.cos(ts), "--", label="y1 exact = cos(t)")
    plt.plot(ts, np.sin(ts), "--", label="y2 exact = sin(t)")
    plt.title("Comparaison des composantes $y_1$ et $y_2$ avec la solution exacte")
    plt.xlabel("t")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()
