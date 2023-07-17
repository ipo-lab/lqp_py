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
    any_eq = n_A > 0
    any_lb = lb.max() > -float("inf")
    any_ub = ub.min() < float("inf")
    any_ineq = any_lb or any_ub

    # --- unpacking control:
    max_iters = control.get('max_iters', 10_000)
    eps_abs = control.get('eps_abs', 1e-3)
    eps_rel = control.get('eps_rel', 1e-3)
    check_solved = control.get('check_solved', max(round((n_x**0.5)/10)*10, 1))
    rho = control.get('rho', None)
    rho_min = control.get('rho_min', 1e-6)
    rho_max = control.get('rho_max', 1e6)
    adaptive_rho = control.get('adaptive_rho', False)
    adaptive_rho_tol = control.get('adaptive_rho_tol', 5)
    adaptive_rho_iter = control.get('adaptive_rho_iter', 100)
    adaptive_rho_iter = round(adaptive_rho_iter / check_solved) * check_solved
    adaptive_rho_max_iter = control.get('adaptive_max_iter', 1000)
    verbose = control.get('verbose', False)
    scale = control.get('scale', False)

    # --- scaling and pre-conditioning:
    if scale:
        D = np.sqrt(1 / np.linalg.norm(Q, ord=np.inf, axis=1))
        Q = D[:, None] * Q * D
        p = D * p
        # --- A scaling:
        if any_eq:
            A = A*D
            E = 1 / np.linalg.norm(A, ord=np.inf, axis=1)
            A = E[:, None] * A
            b = E*b
        if any_ineq:
            lb = lb / D
            ub = ub / D
    else:
        D = 1.0
        E = 1.0

    # --- parameter selection:
    if rho is None:
        if any_eq:
            ATA = np.matmul(A.T, A)
            rho = (np.linalg.norm(Q) / np.linalg.norm(ATA)) ** 0.5
            rho = clamp(rho, rho_min, rho_max)
        else:
            rho = 1.0

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

    # --- main loop
    for i in range(max_iters):
        # --- adaptive rho:
        if adaptive_rho and i % adaptive_rho_iter == 0 and 0 < i < adaptive_rho_max_iter:
            num = primal_error / max(x_norm, z_norm)
            denom = dual_error / y_norm
            denom = clamp(denom, x_min=1e-12)
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
            primal_error = np.linalg.norm(D * r)
            dual_error = np.linalg.norm(D * s)
            if verbose:
                print('iteration = {:d}'.format(i))
                print('|| primal_error||_2 = {:f}'.format(primal_error))
                print('|| dual_error||_2 = {:f}'.format(dual_error))

            x_norm = np.linalg.norm(D * x)
            z_norm = np.linalg.norm(D * z)
            y_norm = np.linalg.norm(rho * D * u)

            tol_primal = eps_abs * n_x ** 0.5 + eps_rel * max(x_norm, z_norm)
            tol_dual = eps_abs * n_x ** 0.5 + eps_rel * y_norm

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



