import torch
from lqp_py.utils import get_ncon, torch_qp_eqcon_mat
from lqp_py.solve_qp_uncon_torch import torch_solve_qp_uncon


def torch_solve_qp_eqcon(Q, p, A, b,):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to Ax =  b
    # Q:  A (n_batch,n_x,n_x) SPD tensor
    # p:  A (n_batch,n_x,1) tensor.
    # A:  A (n_batch,n_eq, n_x) tensor.
    # b:  A (n_batch,n_eq) tensorr.
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################
    # --- prep:
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    if any_eq:
        n_x = p.shape[1]
        # --- setup and solve system
        rhs = torch.cat((-p, b), 1)
        lhs = torch_qp_eqcon_mat(Q=Q, A=A)
        xv = torch.linalg.solve(lhs, rhs)

        # --- unpack solution:
        x = xv[:, :n_x, :]
        nus = xv[:, -n_eq:, :]
        sol = {"x": x, "nus": nus}
    else:
        sol = torch_solve_qp_uncon(Q=Q, p=p)

    return sol


def torch_solve_qp_eqcon_grad(dl_dz, x, nus, Q, A):
    # --- prep:
    n_batch = Q.shape[0]
    dtype = Q.dtype
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0

    # --- sol:
    zeros = torch.zeros((n_batch, n_eq, 1), dtype = dtype)
    sol = torch_solve_qp_eqcon(Q=Q, p=dl_dz, A=A, b=zeros)

    dx = sol.get('x')
    dnu = sol.get('nus')

    dxt = torch.transpose(dx, 1, 2)
    xt = torch.transpose(x, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))

    # --- dl_dA and dl_db
    dl_dA = None
    dl_db = None
    if any_eq:
        dl_db = -dnu
        dl_dA = torch.matmul(dnu, xt) + torch.matmul(nus, dxt)

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db)

    return grads
