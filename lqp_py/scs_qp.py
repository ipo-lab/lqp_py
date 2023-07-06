import torch
import torch.nn as nn
import numpy as np
import scipy
import scs
from lqp_py.utils import get_ncon
from lqp_py.control import scs_control


class SolveScsQP(nn.Module):
    def __init__(self, control=scs_control()):
        super().__init__()
        self.control = control

    def forward(self, Q, p, A, b, G, h):
        x = SolveScsQPLayer.apply(Q, p, A, b, G, h, self.control)
        return x


class SolveScsQPLayer(torch.autograd.Function):
    """
  Autograd function for forward solving and backward differentiating linear constraint QP
  """

    @staticmethod
    def forward(ctx, Q, p, A, b, G, h, control):
        """
    SCS algorithm for forward solving box constraint QP
    """

        # --- forward solve
        sol = torch_solve_qp_scs(Q=Q, p=p, A=A, b=b, G=G, h=h, control=control)
        x = sol.get('x')
        lams = sol.get('lams')
        slacks = sol.get('slacks')

        # --- save for backwards:
        ctx.save_for_backward(x, lams, slacks, Q, A, G)

        return x

    @staticmethod
    def backward(ctx, dl_dx):
        """
    Fixed point backward differentiation
    """
        x, lams, slacks, Q, A, G = ctx.saved_tensors
        grads = torch_solve_qp_scs_grads(dl_dx, x=x, lams=lams, slacks=slacks, Q=Q, A=A, G=G)
        return grads


def torch_solve_qp_scs(Q, p, A, b, G, h, control=scs_control()):
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

    # --- prep:
    n_batch = Q.shape[0]
    n_x = Q.shape[1]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    n_con = n_eq + n_ineq

    if any_eq:
        Amat = torch.cat((A, G), 1)
        bvec = torch.cat((b, h), 1)
    else:
        Amat = G
        bvec = h

    # --- convert to arrays:
    Q_a = torch.asarray(Q).numpy()
    p_a = torch.asarray(p).numpy()
    Amat_a = torch.asarray(Amat).numpy()
    bvec_a = torch.asarray(bvec).numpy()

    # --- holders
    x = np.zeros((n_batch, n_x, 1))
    lams = np.zeros((n_batch, n_con, 1))
    slacks = np.zeros((n_batch, n_con, 1))

    # --- cone and scs control:
    cone = {"z": n_eq, "l": n_ineq}

    # --- main loop: sequential
    for i in range(n_batch):
        # ---  populate dicts with data to pass into SCS
        data = dict(P=scipy.sparse.csc_matrix(Q_a[i, :, :]),
                    A=scipy.sparse.csc_matrix(Amat_a[i, :, :]),
                    b=bvec_a[i, :, 0], c=p_a[i, :, 0])

        # --- initialize solver
        solver = scs.SCS(data=data, cone=cone, **control)
        sol = solver.solve()
        x[i, :, 0] = sol.get('x')
        lams[i, :, 0] = sol.get('y')
        slacks[i, :, 0] = sol.get('s')

    # --- convert to tensor:
    x = torch.tensor(x)
    lams = torch.tensor(lams)
    slacks = torch.tensor(slacks)

    # --- make output list:
    out = {"x": x, "lams": lams, "slacks": slacks}

    return out


def torch_solve_qp_scs_grads(dl_dx, x, lams, slacks, Q, A, G):
    # --- prep:
    n_batch = Q.shape[0]
    n_x = Q.shape[1]
    n_eq = get_ncon(A, dim=1)
    any_eq = n_eq > 0
    n_ineq = get_ncon(G, dim=1)
    n_con = n_eq + n_ineq
    idx_x = np.arange(0, n_x)
    idx_con = np.arange(n_x, n_x + n_con)
    idx_eq = np.arange(0, n_eq)
    idx_ineq = np.arange(n_eq, n_con)

    # --- init: w
    w = torch.cat((x, lams - slacks), dim=1)

    # --- if any eq:
    if any_eq:
        Amat = torch.cat((A, G), 1)
    else:
        Amat = G

    # --- M matrix:
    lhs_u = torch.cat((Q, torch.transpose(Amat, 1, 2)), 2)
    lhs_l = torch.cat((-Amat, torch.zeros(n_batch, n_con, n_con)), 2)
    M = torch.cat((lhs_u, lhs_l), 1)
    I = torch.eye(n_x + n_con).unsqueeze(0)

    # --- gc:
    lhs_u = None
    lhs_l = None

    # --- Derivative of euclidean projection operator:
    idx = np.arange(n_x + n_eq, n_x + n_eq + n_ineq)
    w_y = w[:, idx, :]
    D_w_y = 0.5 * (torch.sign(w_y) + 1)
    ones = torch.ones((n_batch, n_x + n_eq, 1))
    D = torch.cat((ones, D_w_y), 1)

    # --- Core system of Equations:
    rhs = torch.cat((-dl_dx, torch.zeros((n_batch, n_con, 1))), dim=1)
    rhs = D * rhs
    mat = M * torch.transpose(D, 1, 2) - torch.diag_embed(D.squeeze(2)) + I + 10 ** -8 * I

    d = torch.linalg.solve(torch.transpose(mat, 1, 2), rhs)
    # --- gc:
    mat = None

    # --- d:
    dx = d[:, idx_x, :]
    dy = d[:, idx_con, :]

    # --- gradients:
    xt = torch.transpose(x, 1, 2)
    dxt = torch.transpose(dx, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    #dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))
    dl_dQ1 = torch.matmul(0.50 * dx, xt)
    dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)

    # --- dl_dAmat:
    dl_dAmat = torch.matmul(lams, dxt) - torch.matmul(dy, xt)

    dl_dA = None
    dl_db = None
    if any_eq:
        dl_dA = dl_dAmat[:, idx_eq, :]
        dl_db = dy[:, idx_eq, :]
        dl_dG = dl_dAmat[:, idx_ineq, :]
        dl_dh = dy[:, idx_ineq, :]
    else:
        dl_dh = dy
        dl_dG = dl_dAmat

    # --- out list of grads
    grads = (dl_dQ, dl_dp, dl_dA, dl_db, dl_dG, dl_dh, None)
    return grads
