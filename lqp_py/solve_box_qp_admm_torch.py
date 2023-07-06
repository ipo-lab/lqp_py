import torch
import torch.nn as nn
import numpy as np
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

        # --- save for backwards:
        ctx.rho = control.get('rho')
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
            grads = torch_solve_box_qp_grad_kkt(dl_dz, x=x,  lams=lams, nus=nus, Q=Q, A=A, lb=lb, ub=ub)
        else:
            grads = torch_solve_box_qp_grad(dl_dz, x=x, u=u, lams=lams, nus=nus, Q=Q, A=A, lb=lb, ub=ub, rho=rho)
        return grads


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

    # --- unpacking control:
    max_iters = control.get('max_iters', 1000)
    eps_abs = control.get('eps_abs', 0.001)
    eps_rel = control.get('eps_rel', 0.001)
    check_termination = control.get('check_termination', 1)
    rho = control.get('rho', 1)
    verbose = control.get('verbose', False)
    scaling_iter = control.get('scaling_iter', 0)
    aa_iter = control.get('aa_iter', 0)
    reduce = control.get('reduce', 'mean')
    unroll = control.get('unroll', False)

    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    any_lb = torch.max(lb) > -float("inf")
    any_ub = torch.min(ub) < float("inf")
    any_ineq = any_lb or any_ub
    n_x = p.shape[1]
    idx_x = np.arange(0, n_x)
    idx_eq = np.arange(n_x, n_x + n_eq)

    #  --- if not any inequality constraints - zero will solve in a single iteration.
    if not any_ineq:
        rho = 0

    # --- LU factorization:
    Id = torch.eye(n_x).unsqueeze(0)
    M = Q + rho * Id
    if any_eq:
        zero = torch.zeros(n_batch, n_eq, n_eq)
        M1 = torch.cat((M, torch.transpose(A, 1, 2)), 2)
        M2 = torch.cat((A, zero), 2)
        M = torch.cat((M1, M2), 1)

    with torch.no_grad():
        LU, P = torch.linalg.lu_factor(M)  # torch.linalg.lu_solve
    if unroll:
        LUModel = TorchLU(A=M, LU=LU, P=P)

    x = torch.zeros((n_batch, n_x, 1))
    z = torch.zeros((n_batch, n_x, 1))
    u = torch.zeros((n_batch, n_x, 1))
    # --- main loop
    for i in range(max_iters):
        # --- projection to sub-space:
        y = z - u
        if any_eq:
            rhs = torch.cat((-p + rho * y, b), 1)
        else:
            rhs = -p + rho * y

        if unroll:
            xv = LUModel(A=M, b=rhs)
        else:
            xv = torch.linalg.lu_solve(LU, P, rhs)
        x = xv[:, idx_x, :]

        # --- proximal projection:
        z_prev = z
        z = x + u
        if any_ineq:
            z = torch_proj_box(z,
                               lb=lb,
                               ub=ub,
                               any_lb=any_lb,
                               any_ub=any_ub)
        if rho == 0:
            z_prev = z

        # --- andersen acceleration:
        # if aa_iter > 0:
        # --- placeholder for andersen acceleration

        # --- update residuals
        r = x - z
        s = rho * (z - z_prev)
        # --- running sum of residuals or dual variables
        u = u + r

        # ---  primal and dual errors:
        if i % check_termination == 0:
            primal_error = torch.linalg.norm(r, dim=1)
            dual_error = torch.linalg.norm(s, dim=1)
            if reduce == 'mean':
                primal_error = primal_error.mean()
                dual_error = dual_error.mean()
            else:
                primal_error = primal_error.max()
                dual_error = dual_error.max()
            if verbose:
                print('iteration = {:d}'.format(i))
                print('|| primal_error||_2 = {:f}'.format(primal_error.item()))
                print('|| dual_error||_2 = {:f}'.format(dual_error.item()))

            x_norm = torch.linalg.norm(x, dim=1).mean()
            z_norm = torch.linalg.norm(z, dim=1).mean()
            y_norm = torch.linalg.norm(y, dim=1).mean()

            tol_primal = eps_abs * n_x ** 0.5 + eps_rel * max(x_norm, z_norm)
            tol_dual = eps_abs * n_x ** 0.5 + eps_rel * y_norm

            do_stop = primal_error < tol_primal and dual_error < tol_dual
            if do_stop:
                break

    # --- residuals can be computed using x-z, z-z_prev
    lams = u * rho
    lams_neg = torch.threshold(-lams, 0, 0)
    lams_pos = torch.threshold(lams, 0, 0)
    lams = torch.cat((lams_neg, lams_pos), 1)

    nus = None
    if any_eq:
        nus = xv[:, idx_eq, :]

    if unroll:
        sol = x
    else:
        sol = {"x": x, "z": z, "u": u, "lams": lams, "nus": nus}

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
    n_batch = Q.shape[0]
    n_x = Q.shape[1]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    idx_x = np.arange(0, n_x)
    idx_eq = np.arange(n_x, n_x + n_eq)

    # --- derivative of the projection operator:
    xt = torch.transpose(x, 1, 2)
    s_x_u = x + u

    dpi_dx = torch.ones((n_batch, n_x, 1))
    dpi_dx[s_x_u > ub] = 0
    dpi_dx[s_x_u < lb] = 0

    # --- dl_dx: chain rule
    dl_dx = dl_dz * dpi_dx

    # --- rhs:
    if any_eq:
        zeros = torch.zeros((n_batch, n_eq, 1))
        rhs = torch.cat((-dl_dx, zeros), dim=1)
    else:
        rhs = -dl_dx

    # --- this section here can be optimized for speed.
    lhs = dpi_dx * Q
    diag = lhs.diagonal(dim1=1, dim2=2) + rho * (1 - dpi_dx.squeeze(2))
    lhs[:, range(n_x), range(n_x)] = diag
    if any_eq:
        bottom_right = torch.zeros((n_batch, n_eq, n_eq))
        AT = torch.transpose(A, 1, 2)
        lhs_u = torch.cat((lhs, dpi_dx * AT), 2)
        lhs_l = torch.cat((A, bottom_right), 2)
        lhs = torch.cat((lhs_u, lhs_l), 1)

    # --- main system solve -- optimize here?
    d_vec_2 = torch.linalg.solve(lhs, rhs)

    # --- from here
    dv = d_vec_2[:, idx_x, :]
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
        dnu = d_vec_2[:, idx_eq, :]
        dl_db = -dnu
        dl_dA = torch.matmul(dnu, xt) + torch.matmul(nus, dvt)

    # -- simple equation from kkt ...
    kkt = -dl_dz - torch.matmul(Q, dv)
    if any_eq:
        kkt = kkt - torch.matmul(torch.transpose(A, 1, 2), dnu)
    div = rho * u
    div[div == 0] = 1

    dlam = kkt / div

    # --- dl_dlb and dl_dub
    dl_dlb = dlam * lams[:, idx_x, :]
    dl_dub = -dlam * lams[:, n_x + idx_x, :]

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
    idx_x = np.arange(0, n_x)
    idx_ineq = np.arange(n_x, n_x + n_ineq)
    idx_eq = np.arange(n_x + n_ineq, n_x + n_ineq + n_eq)

    # --- rhs:
    rhs = torch.cat((-dl_dz, zeros), dim=1)
    back_sol = torch.linalg.solve(sol_mats, rhs)

    # --- unpack solution:
    dx = back_sol[:, idx_x, :]
    dlam = None
    dnu = None
    if n_ineq > 0:
        dlam = back_sol[:, idx_ineq, :]
    if n_eq > 0:
        dnu = back_sol[:, idx_eq, :]

    diff_list = {"dx": dx, "dlam": dlam, "dnu": dnu}
    return diff_list


def torch_qp_int_grads(x, lams, nus, dx, dlam, dnu):
    # --- prep:
    any_eq = not dnu is None
    any_ineq = not dlam is None

    # --- compute gradients
    # --- some prep:
    xt = torch.transpose(x, 1, 2)
    dxt = torch.transpose(dx, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    #dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))
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
        idx_lb = np.arange(0, n_x)
        idx_ub = np.arange(n_x, n_x + n_x)
        dl_dlb = -dl_dh[:, idx_lb, :]
        dl_dub = dl_dh[:, idx_ub, :]
    elif any_lb:
        idx_lb = np.arange(0, n_x)
        dl_dlb = -dl_dh[:, idx_lb, :]
    elif any_ub:
        idx_ub = np.arange(0, n_x)
        dl_dub = dl_dh[:, idx_ub, :]

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dlb, dl_dub, None)

    return grads
