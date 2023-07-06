import torch
import torch.nn as nn
import numpy as np
from lqp_py.utils import get_ncon
from lqp_py.control import optnet_control
from lqp_py.solve_qp_eqcon_torch import torch_solve_qp_eqcon, torch_solve_qp_eqcon_grad


class OptNet(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control

    def forward(self, Q, p, A, b, G, h):
        x = OptNetLayer.apply(Q, p, A, b, G, h, self.control)
        return x


class OptNetLayer(torch.autograd.Function):
    """
    Autograd function for forward solving and backward differentiating box constraint QP
    """

    @staticmethod
    def forward(ctx, Q, p, A, b, G, h, control=optnet_control()):
        """
        ADMM algorithm for forward solving box constraint QP
        """
        # --- forward solve
        sol = torch_solve_qp_optnet(Q=Q, p=p, A=A, b=b, G=G, h=h, control=control)
        x = sol.get('x')
        lams = sol.get('lams')
        slacks = sol.get('slacks')
        nus = sol.get('nus')
        U_Q = sol.get("U_Q")
        U_S = sol.get("U_S")

        # --- save for backwards:
        ctx.save_for_backward(x, lams, slacks,  nus, U_Q, U_S,  Q, A, G)

        return x

    @staticmethod
    def backward(ctx, dl_dz):
        """
        KKT backward differentiation
        """
        x, lams, slacks, nus, U_Q, U_S, Q, A, G = ctx.saved_tensors
        if lams is None:
            grads = torch_solve_qp_eqcon_grad(dl_dz=dl_dz, x=x, nus=nus, Q=Q, A=A)
            grads = grads + (None, None, None) #G, h and control
        else:
            grads = torch_optnet_grads(dl_dz=dl_dz, x=x, lams=lams, slacks=slacks, nus=nus,
                                       Q=Q, A=A, G=G, U_Q=U_Q, U_S=U_S)
        return grads


def torch_solve_qp_optnet(Q, p, A, b, G, h, control=optnet_control()):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to Ax =  b
    #                        Gx <= h
    # Q:  A (n_batch,n_x,n_x) SPD tensor
    # p:  A (n_batch,n_x,1) tensor.
    # A:  A (n_batch,n_eq, n_x) tensor.
    # b:  A (n_batch,n_eq) tensor.
    # G:  A (n_batch,n_eq, n_x) tensor.
    # h: A (n_batch,n_x,1) tensor
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################

    # --- unpacking control:
    max_iters = control.get('max_iters', 1000)
    tol = control.get('tol', 0.001)
    check_termination = control.get('check_termination', 1)
    verbose = control.get('verbose', False)
    reduce = control.get('reduce', 'mean')
    int_reg = control.get('int_reg', 10 ** -6)

    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    any_ineq = n_ineq > 0
    n_x = p.shape[1]

    # --- if only equality constrained:
    if not any_ineq:
        sol = torch_solve_qp_eqcon(Q=Q, p=p, A=A, b=b)
    else:
        # --- initialize kkt factorization:
        mat_factors = torch_qp_int_pre_factor_kkt(Q=Q, G=G, A=A)
        U_Q = mat_factors.get('U_Q')
        U_S = mat_factors.get('U_S')
        R = mat_factors.get('R')

        # --- initialize:
        sol_init = torch_qp_int_init(Q=Q, p=p, G=G, h=h, A=A, b=b,
                                     U_Q=U_Q, U_S=U_S, R=R, int_reg=int_reg)
        # --- unpack init:
        x = sol_init.get('x')  # main x
        s = sol_init.get('s')  # slacks
        z = sol_init.get('z')  # lams
        y = sol_init.get('y')  # nus

        # --- loop prep:
        GT = torch.transpose(G, 1, 2)
        if any_eq:
            AT = torch.transpose(A, 1, 2)
        one_step = torch.ones((n_batch, 1))
        # --- main loop:
        for i in range(max_iters):
            # --- rhs:
            rx = torch.matmul(GT, z) + torch.matmul(Q, x) + p
            rs = z
            rz = torch.matmul(G, x) + s - h
            if any_eq:
                rx = torch.matmul(AT, y) + rx
                ry = torch.matmul(A, x) - b

                # ---  primal and dual errors:
                if i % check_termination == 0:
                    mu = torch.sum(s * z, dim=1) / n_ineq
                    # -- norm scaling here?
                    prim_resid = torch.linalg.norm(ry, dim=1) + torch.linalg.norm(rz, dim=1)
                    dual_resid = torch.linalg.norm(rx, dim=1)
                    resid = (prim_resid + dual_resid) / 2 + mu

                    # ---- stopping tolerances:
                    if reduce == 'mean':
                        error = resid.mean()
                    else:
                        error = resid.max()
                    if verbose:
                        print('iteration = {:d}'.format(i))
                        print('duality gap = {:f}'.format(error.item()))

                    do_stop = error < tol and i > 0
                    if do_stop:
                        break

                d = z / s

                # ---- factorization
                U_S = torch_qp_int_factor_kkt(U_S=U_S, R=R, d=d, n_eq=n_eq, n_ineq=n_ineq, int_reg=int_reg)

                # ---- affine step
                aff_sol = torch_qp_int_solve_kkt(U_Q=U_Q, d=d, G=G, A=A, U_S=U_S,
                                                 rx=rx, rs=rs, rz=rz, ry=ry)
                dx_aff = aff_sol.get('dx')
                ds_aff = aff_sol.get('ds')
                dz_aff = aff_sol.get('dz')
                dy_aff = aff_sol.get('dy')

                # compute centering directions
                z_step = torch_qp_int_get_step(z, dz_aff)
                s_step = torch_qp_int_get_step(s, ds_aff)
                alphas = torch.cat((z_step, s_step, one_step), 1)
                alpha = torch.min(alphas, 1, keepdim=True).values
                alpha = alpha.unsqueeze(2)
                alpha = alpha * 0.999

                s_plus_step = s + alpha * ds_aff
                z_plus_step = z + alpha * dz_aff
                sig = (torch.sum(s_plus_step * z_plus_step, 1) / (torch.sum(s * z, 1))) ** 3

                mu_sig = -mu * sig
                mu_sig = mu_sig.unsqueeze(2)

                non_zero = (mu_sig + ds_aff * dz_aff) / s
                cor_sol = torch_qp_int_solve_kkt(U_Q=U_Q, d=d, G=G, A=A, U_S=U_S,
                                                 rx=rx * 0, rs=non_zero, rz=rz * 0, ry=ry * 0)

                dx = dx_aff + cor_sol.get('dx')
                ds = ds_aff + cor_sol.get('ds')
                dz = dz_aff + cor_sol.get('dz')
                if any_eq:
                    dy = dy_aff + cor_sol.get('dy')
                else:
                    dy = None

                z_step = torch_qp_int_get_step(z, dz)
                s_step = torch_qp_int_get_step(s, ds)
                alphas = torch.cat((z_step, s_step, one_step), 1)
                alpha = torch.min(alphas, 1, keepdim=True).values
                alpha = alpha.unsqueeze(2)
                alpha = alpha * 0.999

                x = x + alpha * dx
                s = s + alpha * ds
                z = z + alpha * dz
                if any_eq:
                    y = y + alpha * dy

        # --- final factorization:
        d = z / s
        U_S = torch_qp_int_factor_kkt(U_S=U_S, R=R, d=d, n_eq=n_eq, n_ineq=n_ineq, int_reg=int_reg)
        lams = torch.clamp(z, 10 ** -8)
        slacks = torch.clamp(s, 10 ** -8)
        if any_eq:
            nus = y
        else:
            nus = None
        sol = {"x": x, "lams": lams, "slacks": slacks, "nus": nus, "U_Q":U_Q, "U_S":U_S}

    return sol


def torch_qp_int_pre_factor_kkt(Q, G, A):
    # S =  [ A Q^{-1} A^T        A Q^{-1} G^T           ]
    #      [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    # S = rbind(cbind(A_invQ_AT,A_invQ_GT),cbind(G_invQ_AT,G%*%invQ_GT +diag(n_ineq)))
    # chol(S)[1,] == U_S[1,]

    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    n_con = n_eq + n_ineq
    GT = torch.transpose(G, 1, 2)
    AT = torch.transpose(A, 1, 2)

    U_Q = torch.linalg.cholesky(Q, upper=True)
    U_S = torch.zeros((n_batch, n_con, n_con))

    # --- ineq:
    invQ_GT = torch.cholesky_solve(GT, U_Q, upper=True)
    R = torch.matmul(G, invQ_GT)

    if any_eq:
        # --- eq
        invQ_AT = torch.cholesky_solve(AT, U_Q, upper=True)
        A_invQ_AT = torch.matmul(A, invQ_AT)
        U11 = torch.linalg.cholesky(A_invQ_AT, upper=True)

        # --- cross product terms
        G_invQ_AT = torch.matmul(G, invQ_AT)
        U12 = torch.linalg.solve(U11, torch.transpose(G_invQ_AT, 1, 2))

        U1 = torch.cat((U11, U12), dim=2)
        zeros = torch.zeros((n_batch, n_ineq, n_con))
        U_S = torch.cat((U1, zeros), dim=1)

        R = R - torch.matmul(torch.transpose(U12, 1, 2), U12)

    out = {"U_Q": U_Q, "U_S": U_S, "R": R}
    return out


def torch_qp_int_init(Q, p, A, b, G, h, U_Q, U_S, R, int_reg=10 ** -6):
    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)

    # --- rhs:
    d = torch.ones((n_batch, n_ineq, 1))
    rx = p
    rz = -h
    if any_eq:
        ry = -b
    rs = torch.zeros((n_batch, n_ineq, 1))

    # --- factor:
    U_S = torch_qp_int_factor_kkt(U_S=U_S, R=R, d=d, n_eq=n_eq, n_ineq=n_ineq, int_reg=int_reg)

    # --- solve:
    kkt_sol = torch_qp_int_solve_kkt(U_Q=U_Q, d=d, G=G, A=A,
                                     U_S=U_S, rx=rx, rs=rs, rz=rz, ry=ry)

    x = kkt_sol.get('dx')
    s = kkt_sol.get('ds')
    z = kkt_sol.get('dz')
    y = kkt_sol.get('dy')

    # --- set lambdas and slacks positive:
    min_s = torch.min(s, 1).values
    min_z = torch.min(z, 1).values
    s = s + torch.threshold_(1 - min_s, 1, 0).unsqueeze(2)
    z = z + torch.threshold_(1 - min_z, 1, 0).unsqueeze(2)

    # --- sol
    sol = {"x": x, "s": s, "z": z, "y": y}

    return sol


def torch_qp_int_factor_kkt(U_S, R, d, n_eq, n_ineq, int_reg=10 ** -6):
    n_batch = U_S.shape[0]
    zeros = torch.zeros(n_batch, n_ineq, n_eq)

    # --- update U_S for d:
    d_diag = torch.diag_embed(1 / d.squeeze(2))

    reg = torch.eye(n_ineq).unsqueeze(0)
    mat = torch.linalg.cholesky(R + d_diag + int_reg * reg, upper=True)
    mat = torch.cat((zeros, mat), dim=2)

    U1 = U_S[:, [0], :]
    out = torch.cat((U1, mat), dim=1)

    return out


def torch_qp_int_solve_kkt(U_Q, d, G, A, U_S, rx, rs, rz, ry):
    # --- prep:
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    n_con = n_eq + n_ineq
    GT = torch.transpose(G, 1, 2)
    AT = torch.transpose(A, 1, 2)

    invQ_rx = torch.cholesky_solve(rx, U_Q, upper=True)

    # --- ineq and eq con:
    H = torch.matmul(G, invQ_rx) + rs / d - rz
    if any_eq:
        H1 = torch.matmul(A, invQ_rx) - ry
        H = torch.cat((H1, H), dim=1)
    w = -torch.cholesky_solve(H, U_S, upper=True)

    # --- g1:
    if any_eq:
        idx = np.arange(n_eq)
        n_idx = np.arange(n_eq, n_con)
        w_idx = w[:, idx, :, ]
        w_n_idx = w[:, n_idx, :]
        g1 = -rx - torch.matmul(GT, w_n_idx)
        g1 = g1 - torch.matmul(AT, w_idx)
    else:
        w_n_idx = w
        g1 = -rx - torch.matmul(GT, w_n_idx)

    # --- g2
    g2 = -rs - w_n_idx

    dx = torch.cholesky_solve(g1, U_Q, upper=True)
    ds = g2 / d
    dz = w_n_idx
    if any_eq:
        dy = w_idx
    else:
        dy = None

    sol = {"dx": dx, "ds": ds, "dz": dz, "dy": dy}
    return sol


def torch_qp_int_get_step(v, dv):
    a = -v / dv
    z = torch.threshold_(a, 0, float("inf"))
    step = torch.min(z, 1, keepdim=False).values
    return step


def torch_optnet_grads(dl_dz, x, lams, slacks, nus, Q, A, G, U_Q, U_S):
    # --- prep:
    n_batch = Q.shape[0]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    any_ineq = n_ineq > 0

    # --- solve system
    rx = dl_dz
    rs = torch.zeros(n_batch, n_ineq, 1)
    rz = torch.zeros(n_batch, n_ineq, 1)
    if any_eq:
        ry = torch.zeros(n_batch, n_eq, 1)

    d = lams / slacks

    # --- solve kkt system
    diff_list = torch_qp_int_solve_kkt(U_Q=U_Q, d=d, G=G, A=A, U_S=U_S,
                                       rx=rx, rs=rs, rz=rz, ry=ry)
    dx = diff_list.get('dx')
    #ds = diff_list.get('ds')
    dlam = diff_list.get('dz')
    dnu = diff_list.get('dy')

    # --- As per Amos page 5, diff_list$dz = d_lam_tilde = D(lambdas)d_lambda
    dlam = dlam / lams

    # --- compute gradients
    # --- some prep:
    xt = torch.transpose(x, 1, 2)
    dxt = torch.transpose(dx,1,2)

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
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dG, dl_dh, None)
    return grads


