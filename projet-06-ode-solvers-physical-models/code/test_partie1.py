import numpy as np
import matplotlib.pyplot as plt
import partie1

# ==================================================================================
# Test dimension 1 : y'(t) = y(t) / (1 + t^2), y(0) = 1
# ==================================================================================


def f1(t, y):
    return y / (1 + t**2)


def exact1(t):
    return np.exp(np.arctan(t))


def test_1D_n_step():
    y0 = 1.0
    t0 = 0.0
    tf = 5.0
    N = 100
    h = (tf - t0) / N

    ts, ys = partie1.meth_n_step(y0, t0, N, h, f1, partie1.step_midpoint)

    plt.plot(ts, ys, label="Middle Point")
    plt.plot(ts, exact1(ts), "--", label="Exact")
    plt.title("EDO 1D: y' = y / (1 + t^2)")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()


def test_1D_epsilon():
    y0 = 1.0
    t0 = 0.0
    tf = 5.0
    eps = 1e-5

    y_approx = partie1.meth_epsilon(y0, t0, tf, eps, f1, partie1.step_midpoint)
    N = len(y_approx) - 1
    ts = np.linspace(t0, tf, N + 1)
    y_exact = exact1(ts)

    plt.plot(ts, y_approx, label="Middle Point (ε)")
    plt.plot(ts, y_exact, "--", label="Exact")
    plt.title("EDO 1D avec ε")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Erreur max : {np.max(np.abs(y_approx - y_exact))}")


# ==================================================================================
# Test dimension 2 : y(t) = [y1(t), y2(t)], y'(t) = [-y2(t), y1(t)], y(0) = [1, 0]
# ==================================================================================


def f2(t, y):
    return np.array([-y[1], y[0]])


def exact2(t):
    return np.array([np.cos(t), np.sin(t)])


def test_2D_n_step():
    y0 = np.array([1.0, 0.0])
    t0 = 0.0
    tf = 10.0
    N = 200
    h = (tf - t0) / N

    ts, ys = partie1.meth_n_step(y0, t0, N, h, f2, partie1.step_midpoint)
    y1s = ys[:, 0]
    y2s = ys[:, 1]

    plt.plot(ts, y1s, label="y1 - Middle Point")
    plt.plot(ts, y2s, label="y2 - Middle Point")
    plt.plot(ts, np.cos(ts), "--", label="cos(t) - Exact")
    plt.plot(ts, np.sin(ts), "--", label="sin(t) - Exact")
    plt.title("EDO 2D: y' = [-y2, y1]")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()


def test_2D_epsilon():
    y0 = np.array([1.0, 0.0])
    t0 = 0.0
    tf = 10.0
    eps = 1e-5

    y_approx = partie1.meth_epsilon(y0, t0, tf, eps, f2, partie1.step_midpoint)
    N = len(y_approx) - 1
    ts = np.linspace(t0, tf, N + 1)

    plt.plot(ts, y_approx[:, 0], label="y1 - Middle Point (ε)")
    plt.plot(ts, y_approx[:, 1], label="y2 - Middle Point (ε)")
    plt.plot(ts, np.cos(ts), "--", label="cos(t) - Exact")
    plt.plot(ts, np.sin(ts), "--", label="sin(t) - Exact")
    plt.title("EDO 2D avec ε")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()


test_1D_n_step()
test_1D_epsilon()
test_2D_n_step()
test_2D_epsilon()
