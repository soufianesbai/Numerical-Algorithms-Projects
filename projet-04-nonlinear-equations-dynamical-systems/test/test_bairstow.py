import numpy as np
import sys
sys.path.append('../code')
import bairstow

def test_F_bairstow():
    Q = [1, 1]                
    B_test = 2
    C_test = 3     
    R_test = 5
    S_test = 7  
    P = np.polyadd(np.polymul(Q, [1, B_test, C_test]), [R_test, S_test]) # P(X) = (X+1)(X**2+2X+3) + 5X+7
    assert bairstow.F(P, B_test, C_test) == (7.0, 5.0), f"Test failed for F."
    print("All F-Bairstow tests passed successfully!")

def test_deriv_part():
    # degree 2
    P1 = np.array([1, 2, 1])  # P(X) = X**2 + 2X + 1
    B1 = 1
    C1 = 1
    result1 = bairstow.deriv_part(P1, B1, C1)
    expected1 = (0, -1, -1, 0)
    assert np.allclose(result1, expected1), f"Test for degree 2 failed. Expected {expected1}, got {result1}"

    # degree 3
    P2 = np.array([1, 3, 3, 1])  # P(X) = X**3 + 3X**2 + 3X + 1
    B2 = 1
    C2 = 1
    result2 = bairstow.deriv_part(P2, B2, C2)
    expected2 = (-2, -1, 1, 2)
    assert np.allclose(result2, expected2), f"Test for degree 3 failed. Expected {expected2}, got {result2}"

    # zero coefficients
    P3 = np.array([1, 0, 1, 0])  # P(X) = X^3 + X
    B3 = 2
    C3 = 1
    result3 = bairstow.deriv_part(P3, B3, C3)
    expected3 = (2, -1, -5, -2)
    assert np.allclose(result3, expected3), f"Test without coefficients failed. Expected {expected3}, got {result3}"

    # degree 0
    P4 = np.array([5])  # P(X) = 5
    B4 = 3
    C4 = 2
    result4 = bairstow.deriv_part(P4, B4, C4)
    expected4 = (0, 0, 0, 0)
    assert np.allclose(result4, expected4), f"Test for degree 0 failed. Expected {expected4}, got {result4}"

    print("All derivative tests passed successfully!")

def test_bairstow():
    # real roots
    P1 = np.array([1, -6, 11, -6]) # P(X) = (X-1)(X-2)(X-3) = X^3 - 6X^2 + 11X - 6
    result1 = bairstow.Bairstow(P1, [3.0, 2.0])
    expected1 = [1, 2, 3]
    assert np.allclose(sorted(result1), sorted(expected1), atol=1e-6), f"Test for real roots failed. Expected {expected1}, got {result1}"

    # imaginary roots 
    P2 = np.array([1, -1, 1, -1])  # P(X) = (X-1)(X-i)(X+i) = X^3 - X^2 + X - 1
    result2 = bairstow.Bairstow(P2, [1, 1])
    expected2 = [1, 1j, -1j]
    assert np.allclose(sorted(result2, key=lambda x: (x.real, x.imag)), sorted(expected2, key=lambda x: (x.real, x.imag)), atol=1e-6), f"Test for real roots failed. Expected {expected2}, got {result2}"

    # high degree
    P3 = np.array([1, -4, -20, 50, 139, -46, -120]) # P(X) = (X-1)(X+3)(X-5)(X+1)(X-4)(X+2) = X^6 - 4X^5 - 20X^4 + 50X^3 + 139X^2 - 46X -120
    result3 = bairstow.Bairstow(P3, [1, 1])
    expected3 = [1, -3, 5, -1, 4, -2]
    assert np.allclose(sorted(result3), sorted(expected3), atol=1e-6), f"Test for real roots failed. Expected {expected3}, got {result3}"

    print("All Bairstow tests passed successfully!")

if __name__ == "__main__":
    test_F_bairstow()
    test_deriv_part()
    test_bairstow()

