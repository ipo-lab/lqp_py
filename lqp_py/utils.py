import numpy as np
import torch


def make_matrix(x):
    x = np.asarray(x)
    shape = x.shape
    if len(shape) < 2:
        x = x.reshape(-1, 1)

    return x


def get_ncon(x,dim=0):
    if x is None:
        n_con = 0
    else:
        n_con = x.shape[dim]

    return n_con


def torch_qp_eqcon_mat(Q, A, bottom_right=None):
    if bottom_right is None:
        A_size = A.shape
        bottom_right = torch.zeros((A_size[0], A_size[1], A_size[1]))
    AT = torch.transpose(A, 1, 2)
    lhs_u = torch.cat((Q, AT), 2)
    lhs_l = torch.cat((A, bottom_right), 2)
    lhs = torch.cat((lhs_u, lhs_l), 1)

    return lhs