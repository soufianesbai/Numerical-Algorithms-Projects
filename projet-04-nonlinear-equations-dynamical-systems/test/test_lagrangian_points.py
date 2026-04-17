import numpy as np
import sys
sys.path.append('../code')
import newton_raphson, lagrangian_points

U0 = [1.5, 0]
coef_g1=1
coef_g2=0.01
p1=np.array([0,0])
p2=np.array([1,0])
coef_c=1
force, jacobian = lagrangian_points.total_force(U0,coef_g1=1,coef_g2=0.01,p1=np.array([0,0]),p2=np.array([1,0]),coef_c=1)

print("Force vector at U0:", force)
print("Jacobian matrix at U0:", jacobian)
def f(U):
    coef_g1=1
    coef_g2=0.01
    p1=np.array([0,0])
    p2=np.array([1,0])
    coef_c=1
    force, _ = lagrangian_points.total_force(U,coef_g1,coef_g2,p1,p2,coef_c)
    return force

def J(U):
    coef_g1=1
    coef_g2=0.01
    p1=np.array([0,0])
    p2=np.array([1,0])
    coef_c=1
    _, jacobian = lagrangian_points.total_force(U,coef_g1,coef_g2,p1,p2,coef_c)
    return jacobian


equilibrium = newton_raphson.newton_raphson(f, J, U0)

assert np.allclose(f(equilibrium), [0, 0], atol=1e-6)

print("f(ep) = ", f(equilibrium))

print("LAGRANGIAN POINTS")

l = lagrangian_points.compute_lagrangian_points(coef_g1,coef_g2,p1,p2,coef_c)

for i in range(len(l)):
    print(f"L{i+1} : {l[i]}") 
