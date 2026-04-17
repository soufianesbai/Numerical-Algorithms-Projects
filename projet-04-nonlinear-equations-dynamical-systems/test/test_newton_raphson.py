import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../code')
import newton_raphson

def newton_raphson_convergence(f, J, U0, N=100, epsilon=1e-10, alpha=0.5, beta=0.5):
    U = np.array(U0, dtype=float)
    convergence = []  

    for _ in range(N):
        F = f(U)
        H = J(U)

        norm_F = np.linalg.norm(F)
        convergence.append(norm_F)

        if norm_F < epsilon:
            plt.plot(convergence)
            plt.xlabel("Itérations")
            plt.ylabel("Norme de f(U)")
            #plt.yscale('log')
            plt.title("Convergence de la méthode de Newton-Raphson")
            plt.grid(True)
            plt.show()
            return U
        
        V = np.linalg.solve(H, -F)

        t = 1.0  
        while np.linalg.norm(f(U + t * V)) > norm_F + alpha * t * np.linalg.norm(V):
            t *= beta

        U = U + t * V


def test_newton_raphson():
    """Test the Newton-Raphson method with multiple test cases."""
    
    # Test case 1: Simple polynomial equation f(x) = x^4 - 16
    def f1(x):
        return np.array([x[0]**4 - 16])

    def J1(x):
        return np.array([[4 * x[0] ** 3]])

    U01 = [1.0]
    solution1 = newton_raphson.newton_raphson(f1, J1, U01)
    newton_raphson_convergence(f1, J1, U01)
    residual_1 = f1(solution1)
    assert np.allclose(residual_1, 0, atol=1e-5), f"Test failed for case 1: Residual = {residual_1}"
    
    # Test case 2: System of equations f(x, y) = [x^2 + y^2 - 1, x - y]
    def f2(U):
        x, y = U[0], U[1]
        return np.array([
            x**2 + y**2 - 1,
            x - y
        ])

    def J2(U):
        x, y = U[0], U[1]
        return np.array([
            [2*x, 2*y],
            [1, -1]
        ])

    U02 = [0.5, 0.5]
    solution2 = newton_raphson.newton_raphson(f2, J2, U02)
    newton_raphson_convergence(f2, J2, U02)
    residual_2 = f2(solution2)
    assert np.allclose(residual_2, 0, atol=1e-5), f"Test failed for case 2: Residual = {residual_2}"

    # Test case 3: System of equations f(x, y, z) = [
    #   x^2 + y^2 + z^2 - 9, 
    #   x^2 + 2y^2 - z - 3, 
    #   3x - y + z = 0]
    def f3(U):
        x, y, z = U[0], U[1], U[2]
        return np.array([
            x**2 + y**2 + z**2 - 9,
            x**2 + 2*y**2 - z - 3,
            3*x - y + z
        ])

    def J3(U):
        x, y, z = U[0], U[1], U[2]
        return np.array([
            [2*x, 2*y, 2*z],
            [2*x, 4*y, -1],
            [3, -1, 1]
        ])

    U03 = [1.0, 1.0, 1.0]
    solution3 = newton_raphson.newton_raphson(f3, J3, U03)
    newton_raphson_convergence(f3, J3, U03)
    residual_3 = f3(solution3)
    assert np.allclose(residual_3, 0, atol=1e-5), f"Test failed for case 3: Residual = {residual_3}"

    # Test case 4: 
    def f4(U):
        x, y, z = U[0], U[1], U[2]
        return np.array([
            np.sin(x) + y - 2,
            x * y + z - 3,
            np.exp(x) - y * z
        ])

    def J4(U):
        x, y, z = U[0], U[1], U[2]
        return np.array([
            [np.cos(x), 1, 0],
            [y, x, 1],
            [np.exp(x), -z, -y]
        ])

    U04 = [0.5, 0.5, 0.5]
    solution4 = newton_raphson.newton_raphson(f4, J4, U04)
    newton_raphson_convergence(f4, J4, U04)
    residual_4 = f4(solution4)
    assert np.allclose(residual_4, 0, atol=1e-5), f"Test failed for case 4: Residual = {residual_4}"

    # Test for case 4 with different initial guess
    U04_alt = [1.0, 1.0, 1.0]
    solution4_alt = newton_raphson.newton_raphson(f4, J4, U04_alt)
    residual_4_alt = f4(solution4_alt)
    assert np.allclose(residual_4_alt, 0, atol=1e-5), f"Test failed for case 4 (alternative): Residual = {residual_4_alt}"

    print("All Newton-Raphson tests passed successfully!")

test_newton_raphson()
