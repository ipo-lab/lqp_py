import numpy as np
from lqp_py.utils import make_matrix, get_ncon
from lqp_py.solve_qp_uncon import solve_qp_uncon
from scipy.linalg import lu_solve, lu_factor


class BoxQP:
    def __init__(self, Q, p, A, b, lb, ub, control):
        # --- input space:
        self.Q = Q
        self.p = p
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.control = control

        # --- solution storage:
        self.sol = {}

    def solve(self):
        sol = solve_box_qp(Q=self.Q, p=self.p, A=self.A, b=self.b, lb=self.lb, ub=self.ub, control=self.control)
        self.sol = sol
        x = sol.get('x')
        return x

    def update(self, Q=None, p=None, A=None, b=None, lb=None, ub=None, control=None):
        if Q is not None:
            self.Q = Q
        if p is not None:
            self.p = p
        if A is not None:
            self.A = A
        if b is not None:
            self.b = b
        if lb is not None:
            self.lb = None
        if ub is not None:
            self.ub = None
        if control is not None:
            self.control = control
        return None


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
    p = p[:, 0]
    if A is not None:
        A = make_matrix(A)
    if b is not None:
        b = make_matrix(b)
        b = b[:, 0]

    # --- flags:
    n_x = p.shape[0]
    lb = prep_bound(lb, n_x=n_x, default=-float("inf"))
    lb = lb[:, 0]
    ub = prep_bound(ub, n_x=n_x, default=float("inf"))
    ub = ub[:, 0]
    any_eq = A is not None
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

    # --- prep:
    n_x = p.shape[0]
    n_A = get_ncon(A)
    p_norm = np.linalg.norm(p, ord=np.inf)
    any_eq = n_A > 0
    any_lb = lb.max() > -float("inf")
    any_ub = ub.min() < float("inf")
    any_ineq = any_lb or any_ub

    # --- unpacking control:
    max_iters = control.get('max_iters', 10_000)
    eps_abs = control.get('eps_abs', 1e-3)
    eps_abs = max(eps_abs, 1e-16)
    eps_rel = control.get('eps_rel', 1e-3)
    check_solved = control.get('check_solved', max(round((n_x**0.5)/10)*10, 1))
    rho = control.get('rho', None)
    rho_min = control.get('rho_min', 1e-6)
    rho_max = control.get('rho_max', 1e6)
    adaptive_rho = control.get('adaptive_rho', False)
    adaptive_rho_tol = control.get('adaptive_rho_tol', 5)
    adaptive_rho_iter = control.get('adaptive_rho_iter', 100)
    adaptive_rho_iter = round(adaptive_rho_iter / check_solved) * check_solved
    adaptive_rho_iter = max(adaptive_rho_iter, 1)
    adaptive_rho_max_iter = control.get('adaptive_max_iter', 1000)
    adaptive_rho_threshold = control.get('adaptive_rho_threshold', 1e-5)
    verbose = control.get('verbose', False)
    scale = control.get('scale', False)
    beta = control.get('beta')

    # --- scaling and pre-conditioning:
    if scale:
        Q_norm = np.linalg.norm(Q, ord=np.inf, axis=1)
        if Q_norm.min() <= 0.0:
            Q_norm[Q_norm == 0] = Q_norm.mean()
        D = np.sqrt(1 / Q_norm)
        if beta is None:
            v = np.quantile(D, [0.10, 0.90])
            beta = 1 - v[0] / v[1]
        D = (1 - beta) * D + beta * D.mean()
        Q = D[:, None] * Q * D
        p = D * p
        # --- A scaling:
        if any_eq:
            A = A * D
            A_norm = np.linalg.norm(A, ord=np.inf, axis=1)
            if A_norm.min() <= 0.0:
                A_norm[A_norm == 0] = A_norm.mean()
            E = 1 / A_norm
            A = E[:, None] * A
            b = E*b
        else:
            E = 1.0
        if any_ineq:
            lb = lb / D
            ub = ub / D
    else:
        D = 1.0
        E = 1.0

    # --- parameter selection:
    if rho is None:
        Q_norm = np.linalg.norm(Q)
        rho = Q_norm / np.sqrt(n_x)
        rho = clamp(rho, rho_min, rho_max)

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

    # --- LU factorization:
    M_lu = lu_factor(M)

    # --- init:
    x = np.zeros(n_x)
    z = np.zeros(n_x)
    u = np.zeros(n_x)

    # --- init args:
    primal_error = None
    tol_primal_rel_norm = None
    dual_error = None
    tol_dual_rel_norm = None
    xv = None
    i = 0
    zero_clamp = 1e-16
    # --- main loop
    for i in range(max_iters):
        # --- adaptive rho:
        if adaptive_rho and i % adaptive_rho_iter == 0 and 0 < i < adaptive_rho_max_iter:
            if primal_error > adaptive_rho_threshold or dual_error > adaptive_rho_threshold:
                num = primal_error / tol_primal_rel_norm
                num = clamp(num, x_min=zero_clamp)
                denom = dual_error / tol_dual_rel_norm
                denom = clamp(denom, x_min=zero_clamp)
                ratio = (num / denom) ** 0.5
                if ratio > adaptive_rho_tol or ratio < (1 / adaptive_rho_tol):
                    rho = clamp(rho * ratio, rho_min, rho_max)
                    M[:n_x, :n_x] = Q + rho * Id
                    M_lu = lu_factor(M)

        # --- projection to sub-space:
        rhs[:n_x] = -p + rho * (z - u)
        xv = lu_solve(M_lu, rhs)
        x = xv[:n_x]

        # --- proximal projection:
        z_prev = z
        z = x + u
        if any_ineq:
            if any_lb:
                z = np.maximum(z, lb)
            if any_ub:
                z = np.minimum(z, ub)

        if rho == 0:
            z_prev = z

        # --- update residuals
        r = x - z
        s = rho * (z - z_prev)
        # --- running sum of residuals or dual variables
        u = u + r

        # ---  primal and dual errors:
        if i % check_solved == 0:
            # --- reverse scaling:
            primal_error = np.linalg.norm(D * r,  ord=np.inf)
            dual_error = np.linalg.norm(D * s,  ord=np.inf)
            if verbose:
                print(f'iteration = {i}')
                print(f'|| primal_error|| = {primal_error:.10f}')
                print(f'|| dual_error|| = {dual_error:.10f}')

            x_norm = np.linalg.norm(D * x, ord=np.inf)
            z_norm = np.linalg.norm(D * z, ord=np.inf)
            y_norm = np.linalg.norm(rho * D * u, ord=np.inf)
            Qx_norm = np.linalg.norm(np.matmul(Q, x) / D, ord=np.inf)

            tol_primal_rel_norm = max(x_norm, z_norm, zero_clamp)
            tol_primal = eps_abs + eps_rel * tol_primal_rel_norm
            tol_dual_rel_norm = max(y_norm, Qx_norm, p_norm, zero_clamp)
            tol_dual = eps_abs + eps_rel * tol_dual_rel_norm

            do_stop = primal_error < tol_primal and dual_error < tol_dual
            if do_stop:
                break

    # --- reverse the scaling:
    x = D * x
    z = D * z
    u = u / D
    # --- extract dual variables:
    lam = u * rho
    lam_neg = -np.minimum(lam, 0)
    lam_pos = np.maximum(lam, 0)
    lam = np.concatenate((lam_neg, lam_pos), axis=0)
    # --- equality:
    if any_eq:
        nu = xv[-n_A:] * E
    else:
        nu = None
    sol = {"x": x, "z": z, "u": u, "lam": lam, "nu": nu, "rho": rho,
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


def clamp(x, x_min=-float('inf'), x_max=float('inf')):
    return min(max(x, x_min), x_max)
