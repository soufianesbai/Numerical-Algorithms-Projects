import numpy as np
import sys,inspect
sys.path.append("../code")
import newton_raphson
import random

def F(P, B, C):
    """
    Calculates the coefficients R and S of the remainder of the division of P(X) by X**2 + BX + C.

    Args :
        P : Coefficients of the polynomial P(X) in ascending order of powers.
        B : Coefficient of X in the quadratic divisor X**2 + BX + C.
        C : Constant term in the quadratic divisor X**2 + BX + C.

    Returns :
        Tuple (R, S) where :
            - R is the coefficient of X in the rest of the division.
            - S is the constant term in the rest of the division.
    """
    divisor = np.array([1, B, C])
    R = np.polydiv(P, divisor)[1]
    if len(R) == 2:
        return float(R[1]), float(R[0])
    elif len(R) == 1:
        return 0, float(R[0])
    else:
        return 0, 0

def deriv_part(P, B, C):
    """
    Calculates the partial derivates of the previous function.

    Args :
        P : Coefficients of the polynomial P(X) in ascending order of powers.
        B : Coefficient of X in the quadratic divisor X**2 + BX + C.
        C : Constant term in the quadratic divisor X**2 + BX + C.

    Returns :
        Tuple (P1, P2, P3, P4) where :
            - P1 is the partial derivate of R1 differentiating to C.
            - P2 is the partial derivate of S1 differentiating to C.
            - P3 is the partial derivate of R1 differentiating to B.
            - P4 is the partial derivate of S1 differentiating to B.
            with R1 the coefficient of X in the rest of the division of Q by X**2 + BX + C,
                 S1 the constant term of this rest and Q is the quotient of the division of P by X**2 + BX + C.
    """
    divisor = np.array([1, B, C])
    Q = np.polydiv(P, divisor)[0]
    R1, S1 = F(Q, B, C)
    P1 = -R1
    P2 = -S1
    P3 = B * R1 - S1
    P4 = C * R1
    return P1, P2, P3, P4

def get_initial_guess(P):
    """
    Generates initial estimates (r0, s0) for the quadratic divisor X**2 + rX + s 
    in Bairstow's method.

    Args:
        P : Coefficients of the polynomial P(X) in ascending order of powers.

    Returns:
        list: Initial estimates [r0, s0] for the quadratic divisor coefficients.
    """
    n = len(P) - 1
    if n < 2:
        return [1.0, 1.0]

    a_n = P[-1]
    a_n1 = P[-2] if n >= 1 else 0
    a_n2 = P[-3] if n >= 2 else 0

    r0 = (-a_n1/a_n + random.uniform(-0.5, 0.5)) if a_n != 0 else random.uniform(-2, 2)
    s0 = (-a_n2/a_n + random.uniform(-0.5, 0.5)) if a_n != 0 else random.uniform(-2, 2)

    return [r0, s0]


def Bairstow(P, U0=None, N=100, epsilon=1e-8, max_attempts=5):
    """
    Enhanced Bairstow's method for finding polynomial roots with robustness improvements.

    Args:
        P : Polynomial coefficients in ascending order of powers.
        U0 : Initial guess for (r, s) in quadratic divisor X² + rX + s.
        N : Maximum Newton-Raphson iterations per attempt. Default=100.
        epsilon : Convergence tolerance. Default=1e-8.
        max_attempts : Maximum convergence attempts with random restarts. Default=5.

    Returns:
        list: Sorted roots (ascending real part then imaginary part) with duplicates removed.
    """
    current_P = np.array(P, dtype=np.float64)
    roots = []

    if U0 is None:
        U0 = get_initial_guess(current_P)

    while len(current_P) > 1:
        n = len(current_P) - 1

        # degree 1 or 2
        if n <= 2:
            if n == 2:
                a, b, c = current_P
                delta = b**2 - 4*a*c
                if delta >= 0:
                    roots.extend([(-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a)])
                else:
                    roots.extend([complex(-b, np.sqrt(-delta))/(2*a), complex(-b, -np.sqrt(-delta))/(2*a)])
            elif n == 1:
                roots.append(-current_P[0]/current_P[1])
            break

        # functions for Newton-Raphson
        def f(U):
            B, C = U
            return np.array(F(current_P, B, C))

        def J(U):
            B, C = U
            P1, P2, P3, P4 = deriv_part(current_P, B, C)
            return np.array([[P3, P1], [P4, P2]])

        # convergence Attempts
        attempt = 0
        while attempt < max_attempts:
            U = newton_raphson.newton_raphson(f, J, U0, N, epsilon)

            if U is not None:
                r, s = U
                delta = r**2 - 4*s

                # calcul of the roots
                if delta >= 0:
                    sqrt_d = np.sqrt(delta)
                    root1 = (r + sqrt_d)/2
                    root2 = (r - sqrt_d)/2
                else:
                    sqrt_d = cmath.sqrt(delta)
                    root1 = (r + sqrt_d)/2
                    root2 = (r - sqrt_d)/2

                # check before deflation
                new_P, rem = np.polydiv(current_P, [1, r, s])
                if np.linalg.norm(rem) < 1e-6:
                    roots.extend([root1, root2])
                    current_P = new_P
                    U0 = get_initial_guess(current_P)
                    break

            U0 = [random.uniform(-2, 2), random.uniform(-2, 2)]
            attempt += 1

        if attempt == max_attempts:
            # fallback for real roots
            real_roots = [r.real for r in np.roots(current_P) if abs(r.imag) < 1e-6]
            if real_roots:
                roots.append(real_roots[0])
                current_P, _ = np.polydiv(current_P, [1, -real_roots[0]])
            else:
                roots.extend(np.roots(current_P).tolist())
                break

    # elimination of duplicates
    unique_roots = []
    for r in roots:
        if not any(abs(r - u) < 1e-6 for u in unique_roots):
            unique_roots.append(r)

    return sorted(unique_roots, key=lambda x: (x.real, x.imag))