"""Symbolic generation of a synthetic exact solution and corresponding data."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sympy import Piecewise, diff, exp, lambdify, plot, sqrt, symbols
from sympy.plotting import plot3d
from sympy.printing import ccode

def build_synthetic_problem(alpha, nu):
    """Construct synthetic right-hand side, target state and exact control."""
    check_gradient = False
    ccode_output = False
    plot_graphs = False
    plot_error = False

    ################################################################################

    check_point = np.array([[0.5,0.5]])
    # check_point = np.array([
    #     [0.5, 0.5], [0.5, 0.25 + 3 / 16], [0.2, 0.45], [0.5, 0.75],
    #     [0.6, 0.7], [0.5, 0.3], [0.35, 0.35], [0.6, 0.3],
    # ])
    dir_x = np.array([1.,0.])
    dir_y = np.array([0.,1.])

    ################################################################################

    # alpha = 1e-02 # 1e-02     # 1e-2

    ################################################################################

    center = 1.
    # rho1 = 3/16
    # rho2 = 1/4
    # rho3 = 5/16
    rho1 = 1/32
    rho2 = 0.5*center
    rho3 = 2*rho2-rho1
    n = 9
    # nu = 1e-03

    A = np.zeros((n,n))

    for i in range(0,n):
        # values of the polynomial function
        A[0,i] = rho1**i
        A[1,i] = rho2**i
        A[2,i] = rho3**i
        # first derivative
        A[3,i] = i*rho1**(i-1)
        A[4,i] = i*rho3**(i-1)
        # second derivative
        A[5,i] = i*(i-1)*rho1**(i-2)
        A[6,i] = i*(i-1)*rho3**(i-2)
        # thrid derivative
        A[7,i] = i*(i-1)*(i-2)*rho1**(i-3)
        A[8,i] = i*(i-1)*(i-2)*rho3**(i-3)

    b = np.array([0,1,0,0,0,0,0,0,0])

    coeffs = np.linalg.solve(a=A,b=b)

    # print('Coefficients = ', coeffs)

    def PolyCoefficients(x, coeffs):
        """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        o = len(coeffs)
        # print(f'# This is a polynomial of order {o}.')
        y = 0
        for i in range(o):
            y += coeffs[i]*x**i
        return y

    x = np.linspace(rho1, rho3, 100)

    if plot_graphs:
        plt.plot(x, PolyCoefficients(x, coeffs))
        plt.show()


    x = symbols('x')
    y = symbols('y')

    rho = rho2-rho1
    assert rho == rho3-rho2

    # Eucidean norm

    en = sqrt(x**2+y**2)

    # function Psi:

    # polynomial
    Psi_polynomial = (
        coeffs[0]
        + coeffs[1] * x**1
        + coeffs[2] * x**2
        + coeffs[3] * x**3
        + coeffs[4] * x**4
        + coeffs[5] * x**5
        + coeffs[6] * x**6
        + coeffs[7] * x**7
        + coeffs[8] * x**8
    )

    # Psi_poly = Piecewise(
    #     (0, x < rho1),
    #     (0, x > rho3),
    #     (Psi_polynomial, abs(x - rho2) <= rho),
    #     (0, True),
    # )
    Psi_poly = Piecewise((0, x < rho1), (0, x > rho3), (Psi_polynomial, True))

    if plot_graphs:
        plot(Psi_poly,(x,0,center))

    # bump function
    Psi_bump = Piecewise(
        (exp(1) * exp(-1 / (1 - ((x - rho2) / rho) ** 2)), abs(x - rho2) < rho),
        (0, True),
    )

    if plot_graphs:
        plot(Psi_bump,(x,0,center))

    # Psi = Psi_poly or Psi = Psi_bump?
    Psi = Psi_poly
    # Psi = Psi_bump

    Psi_en = Psi.subs(x,en)

    Psi_l = lambdify(x, Psi, "numpy")
    # print(Psi_l(0))


    # function Phi:

    eps = 0.0001

    Phi_0_1 = Piecewise((-Psi_en*x/en, sqrt(x**2+y**2) > eps), (0,True))
    Phi_0_2 = Piecewise((-Psi_en*y/en, sqrt(x**2+y**2) > eps), (0,True))

    # Phi_0_1 = Piecewise((0, sqrt(x**2+y**2) <= eps), (-Psi_en*x/en,True))
    # Phi_0_2 = Piecewise((0, sqrt(x**2+y**2) <= eps), (-Psi_en*y/en,True))

    Phi_1 = Phi_0_1.subs(x,x-center)
    Phi_1 = Phi_1.subs(y,y-center)
    Phi_2 = Phi_0_2.subs(x,x-center)
    Phi_2 = Phi_2.subs(y,y-center)

    Phi = [Phi_1,Phi_2]

    if ccode_output:
        print(ccode(Phi_1))
        print(ccode(Phi))


    Phi_l = lambdify([x,y], Phi, "numpy")

    Phi_div = diff(Phi_1,x) + diff(Phi_2,y)

    if ccode_output:
        print(ccode(Phi_div))

    if check_gradient:
        import utility

        print("######################################################################")
        print("Check derivatives of Phi:")
        Phi_1_lambda = lambdify([x,y], Phi_1, "numpy")
        Phi_1_x_lambda = lambdify([x,y], diff(Phi_1,x), "numpy")
        utility.check_grad(Phi_1_lambda, Phi_1_x_lambda, check_point, dir_x, plot_error)
        Phi_2_lambda = lambdify([x,y], Phi_2, "numpy")
        Phi_2_y_lambda = lambdify([x,y], diff(Phi_2,y), "numpy")
        utility.check_grad(Phi_2_lambda, Phi_2_y_lambda, check_point, dir_y, plot_error)

    Phi_div_l = lambdify([x,y],Phi_div, "numpy")

    # print(Phi_div_l(0.3,0.35))

    # function p

    p = alpha * Phi_div
    p_l = lambdify([x,y],p, "numpy")

    p_laplacian = diff(p, x, x) + diff(p, y, y)

    if ccode_output:
        print(ccode(p_laplacian))

    if check_gradient:
        import utility

        print("######################################################################")
        print("Check derivatives of p:")
        p_lambda = lambdify([x,y], p, "numpy")
        p_x_lambda = lambdify([x,y], diff(p,x), "numpy")
        print("Partial derivative with respect to x:")
        utility.check_grad(p_lambda, p_x_lambda, check_point, dir_x, plot_error)
        p_y_lambda = lambdify([x,y], diff(p,y), "numpy")
        print("Partial derivative with respect to y:")
        utility.check_grad(p_lambda, p_y_lambda, check_point, dir_y, plot_error)
        p_x_x_lambda = lambdify([x,y], diff(p,x,x), "numpy")
        p_y_y_lambda = lambdify([x,y], diff(p,y,y), "numpy")
        print("Second order partial derivative with respect to x:")
        utility.check_grad(p_x_lambda, p_x_x_lambda, check_point, dir_x, plot_error)
        print("Second order partial derivative with respect to y:")
        utility.check_grad(p_y_lambda, p_y_y_lambda, check_point, dir_y, plot_error)

    # function u

    u = Piecewise(
        (0, sqrt((center - x) ** 2 + (center - y) ** 2) > rho2),
        (1, sqrt((center - x) ** 2 + (center - y) ** 2) <= rho2),
    )
    u_l = lambdify([x,y], u, "numpy")

    if ccode_output:
        print(ccode(u))

    y_fun = -2. * x**2 * (2*center - x)**2 * y**2 * (2*center - y)**2
    y_fun_l = lambdify([x,y], y_fun, "numpy")

    if plot_graphs:
        plot3d(y_fun,(x,0,2),(y,0,2))

    if ccode_output:
        print(ccode(y_fun))

    y_laplacian = diff(y_fun, x, x) + diff(y_fun, y, y)

    if ccode_output:
        print(ccode(y_laplacian))

    y_laplacian_l = lambdify([x,y], y_laplacian, "numpy")
    if plot_graphs:
        plot3d(y_laplacian,(x,0,2),(y,0,2))

    if check_gradient:
        import utility

        print("######################################################################")
        print("Check derivatives of y:")
        y_lambda = lambdify([x,y], y_fun, "numpy")
        y_x_lambda = lambdify([x,y], diff(y_fun,x), "numpy")
        print("Partial derivative with respect to x:")
        utility.check_grad(y_lambda, y_x_lambda, check_point, dir_x, plot_error)
        y_y_lambda = lambdify([x,y], diff(y_fun,y), "numpy")
        print("Partial derivative with respect to y")
        utility.check_grad(y_lambda, y_y_lambda, check_point, dir_y, plot_error)
        y_x_x_lambda = lambdify([x,y], diff(y_fun,x,x), "numpy")
        y_y_y_lambda = lambdify([x,y], diff(y_fun,y,y), "numpy")
        print("Second order partial derivative with respect to x:")
        utility.check_grad(y_x_lambda, y_x_x_lambda, check_point, dir_x, plot_error)
        print("Second order partial derivative with respect to y:")
        utility.check_grad(y_y_lambda, y_y_y_lambda, check_point, dir_y, plot_error)

    # function u_d

    # y_d = y_fun + p_laplacian
    y_d = y_fun + nu * p_laplacian - p

    if ccode_output:
        print(ccode(y_d))

    # function f

    # f = - u - y_laplacian
    f = - nu * y_laplacian + y_fun - u

    if ccode_output:
        print(ccode(f))

    f_l = lambdify([x,y], f, "numpy")
    # print(f_l(0.5,0.5))

    y_d_l = lambdify([x,y], y_d, "numpy")
    # print(y_d_l(0.5,0.5))

    def vecs(coord_y, coord_f):
        vec_y = y_d_l(coord_y[:,0],coord_y[:,1])
        vec_f = f_l(coord_f[:,0],coord_f[:,1])
        return vec_y,vec_f
    
    return f_l, y_d_l, y_fun_l, p_l, u_l


# Backward compatibility for previous API name.
exact_solution = build_synthetic_problem
