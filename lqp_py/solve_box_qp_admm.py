import numpy as np
from lqp_py.utils import make_matrix, get_ncon
from lqp_py.solve_qp_uncon import solve_qp_uncon


def solve_box_qp(Q, p, A=None, b=None, lb=-float("inf"), ub=float("inf"), control=None):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to Ax =  b
    #                        lb <= x <= ub
    # Q:  A (n_x,n_x) SPD matrix
    # p:  A (n_x,1) matrix.
    # A:  A (n_eq, n_x) matrix.
    # b:  A (n_eq,1) matrix.
    # lb:  A (n_x,1) vector
    # ub:  A (n_x,1) vector
    # Returns: x_star:  A (n_x) vector
    #######################################################################

    # ---- prep
    Q = make_matrix(Q)
    p = make_matrix(p)
    if not A is None:
        A = make_matrix(A)
    if not b is None:
        b = make_matrix(b)

    # --- flags:
    n_x = p.shape[0]
    lb = prep_bound(lb, n_x=n_x, default=-float("inf"))
    ub = prep_bound(ub, n_x=n_x, default=float("inf"))
    any_eq = not A is None
    any_lb = lb.max() > -float("inf")
    any_ub = ub.max() < float("inf")
    any_ineq = any_lb or any_ub

    #  --- if not any inequality constraints - zero will solve in a single iteration.
    if not any_ineq:
        control['rho'] = 0

    # ---- unconstrained problem
    if not any_eq and not any_ineq:
        sol = solve_qp_uncon(Q=Q, p=p)
    else:
        sol = solve_box_qp_core(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub, control=control)

    return sol


def solve_box_qp_core(Q, p, A, b, lb, ub, control):
    # --- unpacking control:
    max_iters = control.get('max_iters', 1000)
    eps_abs = control.get('eps_abs', 0.001)
    eps_rel = control.get('eps_rel', 0.001)
    check_termination = control.get('check_termination', 1)
    rho = control.get('rho', 1)
    verbose = control.get('verbose', False)
    scaling_iter = control.get('scaling_iter', 0)
    aa_iter = control.get('aa_iter', 0)

    # --- prep:
    n_A = get_ncon(A)
    any_eq = n_A > 0
    any_lb = lb.max() > -float("inf")
    any_ub = ub.min() < float("inf")
    any_ineq = any_lb or any_ub
    n_x = p.shape[0]
    idx_x = np.arange(0, n_x)

    # --- pre-scaling:
    # if scaling_iter > 0:
    # --- place-holder for scaling:

    # --- LU factorization:
    rhs = -p
    Id = np.identity(n_x)
    M = Q + rho * Id
    if any_eq:
        zero = np.zeros((n_A, n_A))
        M1 = np.concatenate((M, A.T), axis=1)
        M2 = np.concatenate((A, zero), axis=1)
        M = np.concatenate((M1, M2), axis=0)
        rhs = np.concatenate((rhs, b), axis=0)

    M_inv = np.linalg.inv(M)  # # --- LU factorization would be better

    x = np.zeros((n_x, 1))
    z = np.zeros((n_x, 1))
    u = np.zeros((n_x, 1))
    # --- main loop
    for i in range(max_iters):
        # --- projection to sub-space:
        y = z - u
        rhs[idx_x, :] = -p + rho * y
        xv = np.dot(M_inv, rhs)
        x = xv[idx_x, :]

        # --- proximal projection:
        z_prev = z
        z = x + u
        if any_ineq:
            if any_lb:
                idx_lb = z < lb
                z[idx_lb] = lb[idx_lb]
            if any_ub:
                idx_ub = z > ub
                z[idx_ub] = ub[idx_ub]

        if rho == 0:
            z_prev = z

        # --- andersen acceleration:
        # if aa_iter > 0:
        # --- placeholder for andersen acceleration

        # --- update residuals
        r = x - z
        s = rho*(z - z_prev)
        # --- running sum of residuals or dual variables
        u = u + r

        # ---  primal and dual errors:
        if i % check_termination == 0:
            primal_error = np.linalg.norm(r)
            dual_error = np.linalg.norm(s)
            if verbose:
                print('iteration = {:d}'.format(i))
                print('|| primal_error||_2 = {:f}'.format(primal_error))
                print('|| dual_error||_2 = {:f}'.format(dual_error))

            x_norm = np.linalg.norm(x)
            z_norm = np.linalg.norm(z)
            y_norm = np.linalg.norm(y)

            tol_primal = eps_abs * n_x ** 0.5 + eps_rel * max(x_norm, z_norm)
            tol_dual = eps_abs * n_x ** 0.5 + eps_rel * y_norm

            do_stop = primal_error < tol_primal and dual_error < tol_dual
            if do_stop:
                break

    # --- extract dual variables:
    lam = u * rho
    lam_neg = -np.minimum(lam, 0)
    lam_pos = np.maximum(lam, 0)
    lam = np.concatenate((lam_neg, lam_pos), axis=0)
    nu = np.delete(xv, idx_x)
    sol = {"x": x, "z": z, "u": u,
           "lam": lam, "nu": nu,
           "primal_error": primal_error, "dual_error": dual_error,
           "iter": i}

    return sol


def prep_bound(x, n_x, default=None):
    if x is None:
        x = default
    x = make_matrix(x)
    if x.shape[0] < n_x:
        x = x.repeat(n_x)
        x = make_matrix(x)
    return x



