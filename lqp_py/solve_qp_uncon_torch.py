import torch


def torch_solve_qp_uncon(Q, p):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx

    # Q:  A (n_batch,n_x,n_x) SPD matrix
    # p:  A (n_batch,n_x,1) matrix.
    # Returns: x_star:  A (n_x,1) matrix
    #######################################################################
    x = torch.linalg.solve(Q, -p)
    sol = {'x': x}
    return sol


def torch_solve_qp_uncon_grad(dl_dz, x, Q):
    # --- sol:
    sol = torch_solve_qp_uncon(Q=Q, p=dl_dz)

    dx = sol.get('x')
    dxt = torch.transpose(dx, 1, 2)
    xt = torch.transpose(x, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))

    # --- out list of grads
    grads = (dl_dQ, dl_dp)

    return grads
