import torch


def torch_uniform(*size,lower=0, upper=1):
    r = torch.rand(*size)
    r = r*(upper-lower) + lower
    return r


def create_qp_data(n_x, n_batch, n_samples, seed=0, requires_grad=True):
    torch.manual_seed(seed)

    L = torch.randn(n_batch, n_samples, n_x, requires_grad=requires_grad)

    Q = torch.matmul(torch.transpose(L, 1, 2), L)
    Q = Q / n_samples
    p = torch.randn(n_batch, n_x, 1, requires_grad=requires_grad)
    A = torch.ones(n_batch, 1, n_x)
    b = torch.ones(n_batch, 1, 1)

    lb = -torch_uniform(n_batch, n_x, 1,lower=1, upper=2)
    ub = torch_uniform(n_batch, n_x, 1,lower=1, upper=2)

    G = torch.cat((-torch.eye(n_x), torch.eye(n_x)))
    G = G.unsqueeze(0) * torch.ones(n_batch, 1, 1)

    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, b, lb, ub, G, h

