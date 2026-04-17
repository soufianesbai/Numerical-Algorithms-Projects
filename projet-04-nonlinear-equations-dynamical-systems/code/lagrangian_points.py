import numpy as np
import newton_raphson,bairstow

def centrifugal_force(U, k, center):
    """
    Compute the centrifugal force at a point U = [x, y].

    Parameters:
    U : Array containing [x, y]
    k : Coefficient of centrifugal force
    center : Center of the centrifugal force [x0, y0]

    Returns:
    np.ndarray : Force vector [fx, fy]
    np.ndarray : Jacobian matrix
    """
    x, y = U
    x0, y0 = center
    fx = k * (x - x0)
    fy = k * (y - y0)
    
    # Jacobian matrix of the centrifugal force
    jacobian = np.array([[k, 0], [0, k]])
    
    return np.array([fx, fy]), jacobian


def gravitational_force(U, k, center,epsilon=1e-10):
    """
    Compute the gravitational force at a point U = [x, y].

    Parameters:
    U : Array containing [x, y]
    k : Gravitational constant
    center : Gravitational force center [x0, y0]

    Returns:
    np.ndarray : Force vector [fx, fy]
    np.ndarray : Jacobian matrix
    """
    x, y = U
    x0, y0 = center
    dx = x - x0
    dy = y - y0
    r_squared = dx**2 + dy**2
    r_cubed = r_squared**(3/2)
    fx = -k * dx / r_cubed
    fy = -k * dy / r_cubed
    r_five = r_squared**(5/2)

    # Jacobian matrix of the gravitational force
    jacobian = np.array([
        [r_squared - 3 * dx**2, -3 * dx * dy],
        [-3 * dx * dy, r_squared - 3 * dy**2]
    ]) * (-k / r_five)

    return np.array([fx, fy]), jacobian

def total_force(U,coef_g1,coef_g2,p1,p2,coef_c):
    """
    Computes the total force from two gravitational forces and one centrifugal force.

    Parameters:
    U : Array containing [x, y], the point where forces are evaluated
    coef_g1 : Coefficient for gravitational force from primary mass at p1
    coef_g2 : Coefficient for gravitational force from secondary mass at p2
    p1 : position of primary mass
    p2 : position of secondary mass
    coef_c : Coefficient for centrifugal force centered at barycenter
    Returns:
    np.ndarray : Total force vector [fx, fy]
    np.ndarray : Jacobian matrix
    """
    # Gravitational force with coefficient 1 at (0, 0)
    f1, J1 = gravitational_force(U, coef_g1, p1 )
    # Gravitational force with coefficient 0.01 at (1, 0)
    f2, J2 = gravitational_force(U, coef_g2, p2)
   

    # Calculate barycenter (fixed y-component calculation)
    x1, y1 = p1
    x2, y2 = p2
    b_x = (coef_g1 * x1 + coef_g2 * x2) / (coef_g1 + coef_g2)
    b_y = (coef_g1 * y1 + coef_g2 * y2) / (coef_g1 + coef_g2)
   
    
    # Centrifugal force centered at barycenter
    fc, Jc = centrifugal_force(U, coef_c, (b_x,b_y))

    # Total force and Jacobian
    force = f1 + f2 + fc
    jacobian = J1 + J2 + Jc

    return force, jacobian



def compute_lagrangian_points(coef_g1,coef_g2,p1,p2,coef_c):
    """
    Computes the positions of all five Lagrangian points (L1-L5) for a two-body system 
    using Bairstow's method for polynomial root-finding and geometric construction.

    Parameters:
    -----------
    coef_g1 : float
        Gravitational coefficient for primary mass at p1 
    coef_g2 : float 
        Gravitational coefficient for secondary mass at p2 
    p1 : array_like
        [x, y] position of primary mass (M1)
    p2 : array_like  
        [x, y] position of secondary mass (M2)
    coef_c : float
        Coefficient for centrifugal force (typically related to angular velocity)

    Returns:
    --------
    list
        List of five [x, y] coordinate pairs for Lagrangian points:
        [L1, L2, L3, L4, L5]
    """
    mu = coef_g2 / (coef_g1 + coef_g2)
    R = np.linalg.norm(p2 - p1)
     
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1


    #Initial x guess
    x0 = [0,0,0]
    x0[0] =  x1+ (1 - (mu/3)**(1/3)) * R 
    x0[1] = x2 + (mu/3)**(1/3) * R
    x0[2] = x1 - (1 + (5/12)*mu) * R
    U0 = [[0,0],[0,0],[0,0]]
    U0[0] = [x0[0],0]
    U0[1] = [x0[1],0]
    U0[2] = [x0[2],0]
    #Computing
    l = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    angle = np.pi / 3 
    
    coeffs = [0,0,0]
    coeffs[0] = np.array([1, mu - 3, 3 - 2*mu, -mu, 2*mu, -mu])
    coeffs[1] = np.array([1, 3 - mu, 3 - 2*mu, -mu, -2*mu, -mu])
    coeffs[2] = np.array([1, 7*mu - 3, -3*mu + 3, -8*mu**2 + 9*mu - 3, 
                  14*mu**2 - 12*mu + 3, -7*mu**2 + 6*mu - 1])

    #L1 L2 L3
    for i in range(3):
        roots = bairstow.Bairstow(coeffs[i])  
        for root in roots:
            #Taking the first real root
            if not np.iscomplex(root):
                l[i][0] = root
                break
            #No real root
            else:  
                l[i][0] = roots[0]  
    #L4
    l[3][0] = x1 + dx * np.cos(angle) - dy * np.sin(angle)
    l[3][1] = y1 + dx * np.sin(angle) + dy * np.cos(angle)
    
    #L5 
    l[4][0] = x1 + dx * np.cos(angle) + dy * np.sin(angle)
    l[4][1] = y1 - dx * np.sin(angle) + dy * np.cos(angle)

    return l