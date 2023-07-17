import numpy as np
import torch


def torch_uniform(*size, lower=0, upper=1):
    r = torch.rand(*size)
    r = r * (upper - lower) + lower
    return r


def create_qp_data(n_x, n_batch, n_samples, seed=0, requires_grad=True):
    torch.manual_seed(seed)

    L = torch.randn(n_batch, n_samples, n_x)

    Q = torch.matmul(torch.transpose(L, 1, 2), L)
    Q = Q / n_samples
    Q.requires_grad = requires_grad
    p = torch.randn(n_batch, n_x, 1, requires_grad=requires_grad)
    A = torch.ones(n_batch, 1, n_x)
    b = torch.ones(n_batch, 1, 1)

    lb = -torch_uniform(n_batch, n_x, 1, lower=1, upper=2)
    ub = torch_uniform(n_batch, n_x, 1, lower=1, upper=2)

    G = torch.cat((-torch.eye(n_x), torch.eye(n_x)))
    G = G.unsqueeze(0) * torch.ones(n_batch, 1, 1)

    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, b, lb, ub, G, h


def generate_hard_qp(n_x, prob, seed):
    np.random.seed(seed)
    M = np.random.normal(size=(n_x, n_x))
    M = M * np.random.binomial(1, prob, size=(n_x, n_x))
    # --- Q:
    Q = np.dot(M.T, M) + 1e-2 * np.eye(n_x)
    # --- p:
    p = np.random.normal(size=(n_x, 1))
    # --- x0:
    x0 = np.random.normal(size=(n_x, 1))
    s_lb = -np.random.uniform(size=(n_x, 1))
    s_ub = np.random.uniform(size=(n_x, 1))
    # --- A:
    A = generate_random_A(n_x=n_x, prob=prob)
    # --- b:
    b = np.matmul(A, x0)
    # --- ub:
    ub = x0 + s_ub
    # --- lb:
    lb = x0 + s_lb
    return Q, p, A, b, lb, ub


def generate_hard_qp_torch(n_x, prob, seeds, dtype=torch.float64):
    n_batch = len(seeds)
    m = round(n_x ** 0.5)
    Q = np.zeros((n_batch, n_x, n_x))
    p = np.zeros((n_batch, n_x, 1))
    A = np.zeros((n_batch, m, n_x))
    b = np.zeros((n_batch, m, 1))
    lb = np.zeros((n_batch, n_x, 1))
    ub = np.zeros((n_batch, n_x, 1))
    for i in range(n_batch):
        Q_i, p_i, A_i, b_i, lb_i, ub_i = generate_hard_qp(n_x=n_x, prob=prob, seed=seeds[i])
        Q[i, :, :] = Q_i
        p[i, :, :] = p_i
        A[i, :, :] = A_i
        b[i, :, :] = b_i
        lb[i, :, :] = lb_i
        ub[i, :, :] = ub_i

    Q = torch.tensor(Q, dtype=dtype, requires_grad=True)
    p = torch.tensor(p, dtype=dtype,  requires_grad=True)
    A = torch.tensor(A, dtype=dtype,  requires_grad=True)
    b = torch.tensor(b, dtype=dtype,  requires_grad=True)
    lb = torch.tensor(lb, dtype=dtype,  requires_grad=True)
    ub = torch.tensor(ub, dtype=dtype,  requires_grad=True)
    # --- add G, h
    G = torch.cat((-torch.eye(n_x, dtype=dtype), torch.eye(n_x, dtype=dtype)))
    G = G.unsqueeze(0) * torch.ones(n_batch, 1, 1, dtype=dtype)
    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, b, lb, ub, G, h


def generate_random_A(n_x, prob):
    # --- A:
    m = round(n_x ** 0.5)
    A = np.zeros((m, n_x))
    for i in range(m):
        a = np.random.normal(size=(1, n_x))
        b = np.zeros(1)
        while b.sum() == 0:
            b = np.random.binomial(1, prob, size=(1, n_x))
        a = a*b
        A[i, :] = a

    return A
