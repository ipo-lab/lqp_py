import torch
import torch.nn as nn
from lqp_py.utils import get_ncon
from lqp_py.lu_layer import TorchLU


class SolveBoxQP(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control

    def forward(self, Q, p, A, b, lb, ub):
        unroll = self.control.get('unroll', False)
        if unroll:
            x = torch_solve_box_qp(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub, control=self.control)
        else:
            x = SolveBoxQPLayer.apply(Q, p, A, b, lb, ub, self.control)
        return x


class SolveBoxQPLayer(torch.autograd.Function):
    """
    Autograd function for forward solving and backward differentiating box constraint QP
    """

    @staticmethod
    def forward(ctx, Q, p, A, b, lb, ub, control):
        """
        ADMM algorithm for forward solving box constraint QP
        """

        # --- check for ineq:
        any_lb = torch.max(lb) > -float("inf")
        any_ub = torch.min(ub) < float("inf")
        any_ineq = any_lb or any_ub
        #  --- if not any inequality constraints - zero will solve in a single iteration.
        if not any_ineq:
            control['rho'] = 0

        # --- forward solve
        sol = torch_solve_box_qp(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub, control=control)
        x = sol.get('x')
        u = sol.get('u')
        lams = sol.get('lams')
        nus = sol.get('nus')
        rho = sol.get('rho')

        # --- save for backwards:
        ctx.rho = rho
        ctx.backward_method = control.get('backward', 'fixed_point')
        ctx.save_for_backward(x, u, lams, nus, Q, A, lb, ub)

        return x

    @staticmethod
    def backward(ctx, dl_dz):
        """
        Fixed point backward differentiation
        """
        x, u, lams, nus, Q, A, lb, ub = ctx.saved_tensors
        rho = ctx.rho
        backward_method = ctx.backward_method
        if backward_method == 'kkt':
            grads = torch_solve_box_qp_grad_kkt(dl_dz, x=x, lams=lams, nus=nus, Q=Q, A=A, lb=lb, ub=ub)
        else:
            grads = torch_solve_box_qp_grad(dl_dz, x=x, u=u, lams=lams, nus=nus, Q=Q, A=A, lb=lb, ub=ub, rho=rho)
        return grads


class BoxQPTH:
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
        sol = torch_solve_box_qp(Q=self.Q, p=self.p, A=self.A, b=self.b, lb=self.lb, ub=self.ub, control=self.control)
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


def torch_solve_box_qp(Q, p, A, b, lb, ub, control):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to Ax =  b
    #                        lb <= x <= ub
    # Q:  A (n_batch,n_x,n_x) SPD tensor
    # p:  A (n_batch,n_x,1) tensor.
    # A:  A (n_batch,n_eq, n_x) tensor.
    # b:  A (n_batch,n_eq) tensorr.
    # lb: A (n_batch,n_x,1) tensor
    # ub: A (n_batch,n_x,1) tensor
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################

    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    n_x = p.shape[1]
    p_norm = torch.linalg.norm(p, ord=torch.inf, dim=1, keepdim=True)
    any_eq = n_eq > 0
    any_lb = torch.max(lb) > -float("inf")
    any_ub = torch.min(ub) < float("inf")
    any_ineq = any_lb or any_ub

    # --- unpacking control:
    max_iters = control.get('max_iters', 10_000)
    eps_abs = control.get('eps_abs', 1e-3)
    eps_abs = max(eps_abs, 1e-12)
    eps_rel = control.get('eps_rel', 1e-3)
    eps_rel = max(eps_rel, 1e-12)
    check_solved = control.get('check_solved', max(round((n_x ** 0.5) / 10) * 10, 1))
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
    adaptive_rho_threshold = torch.ones(1) * adaptive_rho_threshold
    verbose = control.get('verbose', False)
    scale = control.get('scale', False)
    beta = control.get('beta')
    unroll = control.get('unroll', False)

    #  --- if not any inequality constraints - zero will solve in a single iteration.
    if not any_ineq:
        rho = 0

    # --- scaling and pre-conditioning:
    if scale:
        # --- Q and p scaling:
        Q_norm = torch.linalg.norm(Q, ord=torch.inf, dim=1)
        idx = Q_norm <= 0.0
        if torch.any(idx):
            Q_norm_min = torch.clamp(Q_norm.mean(dim=1), min=1e-6)
            Q_norm_clamp = torch.clamp(Q_norm, min=Q_norm_min.unsqueeze(1))
            Q_norm[idx] = Q_norm_clamp[idx]
        # --- compute D:
        D = torch.sqrt(1 / Q_norm)
        if beta is None:
            v = torch.quantile(D, q=torch.tensor([0.10, 0.90], dtype=D.dtype), dim=1)
            beta = 1 - v[[0]] / v[[1]]
            beta = beta.T
        D = (1 - beta) * D + beta * D.mean(dim=1, keepdim=True)
        Q = (D.unsqueeze(2) * Q * D.unsqueeze(1))
        p = D.unsqueeze(2) * p
        # --- A scaling:
        if any_eq:
            A = A * D.unsqueeze(1)
            A_norm = torch.linalg.norm(A, ord=torch.inf, dim=2)
            idx = A_norm <= 0.0
            if torch.any(idx):
                AD_norm_min = torch.clamp(A_norm.mean(dim=1), min=1e-6)
                AD_norm_clamp = torch.clamp(A_norm, min=AD_norm_min.unsqueeze(1))
                A_norm[idx] = AD_norm_clamp[idx]
            E = 1 / A_norm
            E = E.unsqueeze(2)
            A = E * A
            b = E * b
        D = D.unsqueeze(2)
        if any_ineq:
            lb = lb / D
            ub = ub / D
    else:
        D = 1.0
        E = 1.0

    # --- rho parameter selection:
    if rho is None:
        Q_norm = torch.linalg.matrix_norm(Q, keepdim=True)
        rho = Q_norm / n_x ** 0.5
        rho = torch.clamp(rho, min=rho_min, max=rho_max)

    # --- LU factorization:
    Id = torch.eye(n_x, dtype=p.dtype).unsqueeze(0)
    M = Q + rho * Id
    if any_eq:
        zero = torch.zeros(n_batch, n_eq, n_eq, dtype=p.dtype)
        M1 = torch.cat((M, torch.transpose(A, 1, 2)), 2)
        M2 = torch.cat((A, zero), 2)
        M = torch.cat((M1, M2), 1)

    with torch.no_grad():
        LU, P = torch.linalg.lu_factor(M)  # torch.linalg.lu_solve
    if unroll:
        LUModel = TorchLU(A=M, LU=LU, P=P)
    else:
        LUModel = None
    # --- init solutions:
    x = torch.zeros((n_batch, n_x, 1), dtype=p.dtype)
    z = torch.zeros((n_batch, n_x, 1), dtype=p.dtype)
    u = torch.zeros((n_batch, n_x, 1), dtype=p.dtype)
    # --- init:
    primal_error = None
    tol_primal_rel_norm = None
    dual_error = None
    tol_dual_rel_norm = None
    zero_clamp = 1e-16
    eps_rel_tensor = torch.ones(1) * zero_clamp
    xv = None
    do_rho_update = adaptive_rho
    i = 0
    # --- main loop
    for i in range(max_iters):
        # --- adaptive rho:
        if adaptive_rho and i % adaptive_rho_iter == 0 and 0 < i < adaptive_rho_max_iter:
            if torch.any(do_rho_update):
                num = primal_error / tol_primal_rel_norm
                num = torch.clamp(num, min=zero_clamp)
                denom = dual_error / tol_dual_rel_norm
                denom = torch.clamp(denom, min=zero_clamp)
                ratio = (num / denom) ** 0.5
                update_rho_1 = (ratio > adaptive_rho_tol).sum() > 0
                update_rho_2 = (ratio < (1 / adaptive_rho_tol)).sum() > 0
                update_rho = update_rho_1.item() or update_rho_2.item()
                if update_rho:
                    rho_new = rho * ratio
                    rho = rho * torch.logical_not(do_rho_update) + rho_new * do_rho_update
                    rho = torch.clamp(rho, min=rho_min, max=rho_max)
                    # --- note we should be able to just update LU directly as only diagonal is changing.
                    M[:, :n_x, :n_x] = Q + rho * Id
                    with torch.no_grad():
                        LU, P = torch.linalg.lu_factor(M)
                    if unroll:
                        LUModel = TorchLU(A=M, LU=LU, P=P)

        # --- projection to sub-space:
        if any_eq:
            rhs = torch.cat((-p + rho * (z - u), b), 1)
        else:
            rhs = -p + rho * (z - u)

        if unroll:
            xv = LUModel(A=M, b=rhs)
        else:
            xv = torch.linalg.lu_solve(LU, P, rhs)
        x = xv[:, :n_x, :]

        # --- proximal projection:
        z_prev = z
        z = x + u
        if any_lb:
            z = torch.maximum(z, lb)
        if any_ub:
            z = torch.minimum(z, ub)

        # --- update residuals
        r = x - z
        s = rho * (z - z_prev)
        # --- running sum of residuals or dual variables
        u = u + r

        # ---  primal and dual errors:
        if i % check_solved == 0:
            primal_error = torch.linalg.norm(D * r, ord=torch.inf, dim=1, keepdim=True)
            dual_error = torch.linalg.norm(D * s, ord=torch.inf, dim=1, keepdim=True)

            if verbose:
                primal_error_max = primal_error.max()
                dual_error_max = dual_error.max()
                print(f'iteration = {i}')
                print(f'|| primal_error|| = {primal_error_max.item():.10f}')
                print(f'|| dual_error|| = {dual_error_max.item():.10f}')

            x_norm = torch.linalg.norm(D * x, ord=torch.inf, dim=1, keepdim=True)
            z_norm = torch.linalg.norm(D * z, ord=torch.inf, dim=1, keepdim=True)
            y_norm = torch.linalg.norm(rho * D * u, ord=torch.inf, dim=1, keepdim=True)
            Qx_norm = torch.linalg.norm(torch.matmul(Q, x) / D, ord=torch.inf, dim=1, keepdim=True)

            tol_primal_rel_norm = torch.maximum(torch.maximum(x_norm, z_norm), eps_rel_tensor)
            tol_primal = eps_abs + eps_rel * tol_primal_rel_norm
            tol_dual_rel_norm = torch.maximum(torch.maximum(torch.maximum(y_norm, Qx_norm), p_norm), eps_rel_tensor)
            tol_dual = eps_abs + eps_rel * tol_dual_rel_norm

            # --- check for optimality
            do_stop_primal = primal_error < tol_primal
            do_stop_dual = dual_error < tol_dual
            is_optimal = torch.logical_and(do_stop_primal, do_stop_dual)
            do_rho_update = torch.logical_or(primal_error > torch.maximum(tol_primal, adaptive_rho_threshold),
                                             dual_error > torch.maximum(tol_dual, adaptive_rho_threshold))
            if torch.all(is_optimal).item():
                break

    # --- reverse the scaling:
    x = D * x
    z = D * z
    u = u / D
    # --- residuals can be computed using x-z, z-z_prev
    lams = u * rho
    lams_neg = torch.threshold(-lams, 0, 0)
    lams_pos = torch.threshold(lams, 0, 0)
    lams = torch.cat((lams_neg, lams_pos), 1)

    nus = None
    if any_eq:
        nus = xv[:, -n_eq:, :] * E
    if unroll:
        sol = x
    else:
        sol = {"x": x, "z": z, "u": u, "lams": lams, "nus": nus, "rho": rho, "iter": i}

    return sol


def torch_proj_box(x, lb, ub, any_lb=True, any_ub=True):
    if any_lb:
        lb_diff = lb - x
        lb_diff_relu = torch.relu(lb_diff)
        x = x + lb_diff_relu

    if any_ub:
        ub_diff = x - ub
        ub_diff_relu = torch.relu(ub_diff)
        x = x - ub_diff_relu
    return x


def torch_solve_box_qp_grad(dl_dz, x, u, lams, nus, Q, A, lb, ub, rho):
    # --- prep:
    dtype = x.dtype
    n_batch = Q.shape[0]
    n_x = Q.shape[1]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    if rho is None:
        rho = 1.0

    # --- derivative of the projection operator:
    xt = torch.transpose(x, 1, 2)
    s_x_u = x + u

    dpi_dx = torch.ones((n_batch, n_x, 1), dtype=dtype)
    dpi_dx[s_x_u > ub] = 0
    dpi_dx[s_x_u < lb] = 0

    # --- dl_dx: chain rule
    dl_dx = dl_dz * dpi_dx

    # --- rhs:
    if any_eq:
        zeros = torch.zeros((n_batch, n_eq, 1), dtype=dtype)
        rhs = torch.cat((-dl_dx, zeros), dim=1)
    else:
        rhs = -dl_dx

    # --- this section here can be optimized for speed.
    lhs = dpi_dx * Q
    if torch.is_tensor(rho):
        diag = lhs.diagonal(dim1=1, dim2=2) + rho.squeeze(2) * (1 - dpi_dx.squeeze(2))
    else:
        diag = lhs.diagonal(dim1=1, dim2=2) + rho * (1 - dpi_dx.squeeze(2))
    lhs[:, range(n_x), range(n_x)] = diag
    if any_eq:
        bottom_right = torch.zeros((n_batch, n_eq, n_eq), dtype=dtype)
        AT = torch.transpose(A, 1, 2)
        lhs_u = torch.cat((lhs, dpi_dx * AT), 2)
        lhs_l = torch.cat((A, bottom_right), 2)
        lhs = torch.cat((lhs_u, lhs_l), 1)

    # --- main system solve -- optimize here?
    lhs[:, range(n_x + n_eq), range(n_x + n_eq)] = lhs.diagonal(dim1=1, dim2=2) + 1e-8  # ---small regularizer
    d_vec_2 = torch.linalg.solve(lhs, rhs)

    # --- from here
    dv = d_vec_2[:, :n_x, :]
    dvt = torch.transpose(dv, 1, 2)

    # --- dl_dp
    dl_dp = dv

    # --- dl_dQ dl_dQ = 0.5 * (torch.matmul(dv, xt) + torch.matmul(x, dvt))
    dl_dQ1 = torch.matmul(0.50 * dv, xt)
    dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)

    # --- dl_dA and dl_db
    dl_db = None
    dl_dA = None
    if any_eq:
        dnu = d_vec_2[:, -n_eq:, :]
        dl_db = -dnu
        dl_dA = torch.matmul(dnu, xt) + torch.matmul(nus, dvt)
    else:
        dnu = None

    # -- simple equation from kkt ...
    kkt = -dl_dz - torch.matmul(Q, dv)
    if any_eq:
        kkt = kkt - torch.matmul(torch.transpose(A, 1, 2), dnu)
    div = rho * u
    div[div == 0] = 1

    dlam = kkt / div

    # --- dl_dlb and dl_dub
    dl_dlb = dlam * lams[:, :n_x, :]
    dl_dub = -dlam * lams[:, n_x:(2 * n_x), :]

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dlb, dl_dub, None)

    return grads


def torch_solve_box_qp_grad_kkt(dl_dz, x, lams, nus, Q, A, lb, ub):
    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_lb = torch.max(lb) > -float("inf")
    any_ub = torch.min(ub) < float("inf")
    any_ineq = any_lb or any_ub
    n_x = Q.shape[1]

    # --- make h and G
    G = None
    if any_ineq:
        G = torch.cat((-torch.eye(n_x), torch.eye(n_x)))
        G = G.unsqueeze(0) * torch.ones(n_batch, 1, 1)
        h = torch.cat((-lb, ub), dim=1)
        slacks = h - torch.matmul(G, x)
        slacks = torch.clamp(slacks, 10 ** -8)
        lams = torch.clamp(lams, 10 ** -8)
    else:
        slacks = None
    n_ineq = get_ncon(G, dim=1)

    # --- make inversion matrix:
    sol_mats = torch_qp_make_sol_mat(Q=Q, G=G, A=A, lams=lams, slacks=slacks)

    # --- Compute differentials:
    diff_list = torch_solve_qp_backwards(dl_dz=dl_dz, sol_mats=sol_mats, n_eq=n_eq, n_ineq=n_ineq)
    dx = diff_list.get('dx')
    dlam = diff_list.get('dlam')
    dnu = diff_list.get('dnu')

    # --- compute gradients
    grads = torch_qp_int_grads_admm(x=x, lams=lams, nus=nus, dx=dx, dlam=dlam, dnu=dnu, any_lb=any_lb, any_ub=any_ub)

    return grads


def torch_qp_make_sol_mat(Q, G, A, lams, slacks):
    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    any_ineq = n_ineq > 0
    AT = None
    GT = None
    if any_eq:
        AT = torch.transpose(A, 1, 2)
    if any_ineq:
        GT = torch.transpose(G, 1, 2)

    if not any_eq:
        lhs_1 = torch.cat((Q, GT * torch.transpose(lams, 1, 2)), 2)
        lhs_2 = torch.cat((G, torch.diag_embed(-slacks.squeeze(2))), 2)
        lhs = torch.cat((lhs_1, lhs_2), 1)
    elif not any_ineq:
        lhs_1 = torch.cat((Q, AT), 2)
        lhs_2 = torch.cat((A, torch.zeros(n_batch, n_eq, n_eq)), 2)
        lhs = torch.cat((lhs_1, lhs_2), 1)
    else:
        lhs_1 = torch.cat((Q, GT * torch.transpose(lams, 1, 2), AT), 2)
        lhs_2 = torch.cat((G, torch.diag_embed(-slacks.squeeze(2)), torch.zeros(n_batch, n_ineq, n_eq)), 2)
        lhs_3 = torch.cat((A, torch.zeros(n_batch, n_eq, n_ineq), torch.zeros(n_batch, n_eq, n_eq)), 2)
        lhs = torch.cat((lhs_1, lhs_2, lhs_3), 1)

    return lhs


def torch_solve_qp_backwards(dl_dz, sol_mats, n_eq, n_ineq):
    # --- prep:
    n_batch = dl_dz.shape[0]
    n_x = dl_dz.shape[1]
    n_con = n_eq + n_ineq
    zeros = torch.zeros(n_batch, n_con, 1)

    # --- rhs:
    rhs = torch.cat((-dl_dz, zeros), dim=1)
    back_sol = torch.linalg.solve(sol_mats, rhs)

    # --- unpack solution:
    dx = back_sol[:, :n_x, :]
    dlam = None
    dnu = None
    if n_ineq > 0:
        dlam = back_sol[:, n_x:(n_x + n_ineq), :]
    if n_eq > 0:
        dnu = back_sol[:, (n_x + n_ineq):(n_x + n_ineq + n_eq), :]

    diff_list = {"dx": dx, "dlam": dlam, "dnu": dnu}
    return diff_list


def torch_qp_int_grads(x, lams, nus, dx, dlam, dnu):
    # --- prep:
    any_eq = not isinstance(dnu, type(None))
    any_ineq = not isinstance(dlam, type(None))

    # --- compute gradients
    # --- some prep:
    xt = torch.transpose(x, 1, 2)
    dxt = torch.transpose(dx, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    # dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))
    dl_dQ1 = torch.matmul(0.50 * dx, xt)
    dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)

    # --- inequality
    dl_dG = None
    dl_dh = None
    if any_ineq:
        D_lams = torch.diag_embed(lams.squeeze(2))
        dl_dG = torch.matmul(D_lams, torch.matmul(dlam, xt)) + torch.matmul(lams, dxt)
        dl_dh = -lams * dlam

    # --- equality
    dl_dA = None
    dl_db = None
    if any_eq:
        dl_dA = torch.matmul(dnu, xt) + torch.matmul(nus, dxt)
        dl_db = -dnu

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dG, dl_dh)
    return grads


def torch_qp_int_grads_admm(x, lams, nus, dx, dlam, dnu, any_lb, any_ub):
    # --- prep:
    n_x = x.shape[1]
    # --- compute regular qp gradients:
    dl_dQ, dl_dp, dl_dA, dl_db, dl_dG, dl_dh = torch_qp_int_grads(x=x, lams=lams, nus=nus, dx=dx, dlam=dlam, dnu=dnu)

    dl_dlb = None
    dl_dub = None
    if any_lb & any_ub:
        dl_dlb = -dl_dh[:, :n_x, :]
        dl_dub = dl_dh[:, n_x:(2 * n_x), :]
    elif any_lb:
        dl_dlb = -dl_dh[:, :n_x, :]
    elif any_ub:
        dl_dub = dl_dh[:, :n_x, :]

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dlb, dl_dub, None)

    return grads
