import numpy as np


def solve_qp_uncon(Q, p,):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx

    # Q:  A (n_x,n_x) SPD matrix
    # p:  A (n_eq,1) matrix.
    # Returns: x_star:  A (n_x) matrix
    #######################################################################
    x = np.linalg.solve(Q, -p)
    sol = {"x": x}
    return sol
