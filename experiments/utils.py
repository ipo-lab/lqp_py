import numpy as np
import torch


def torch_uniform(*size,lower=0, upper=1):
    r = torch.rand(*size)
    r = r*(upper-lower) + lower
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


def generate_random_qp(n_x, m, prob, seed):
    np.random.seed(seed)
    M = np.random.normal(size=(n_x, n_x))
    M = M * np.random.binomial(1, prob, size=(n_x, n_x))
    # --- Q:
    Q = np.dot(M.T, M) + 10 ** -2 * np.eye(n_x)
    # --- p:
    p = np.random.normal(size=(n_x, 1))
    # --- A:
    A = generate_random_A(n_x=n_x, m=m, prob=prob)
    # --- b:
    b = np.random.uniform(size=(A.shape[0], 1))
    # --- ub:
    ub = np.random.uniform(size=(n_x, 1))
    # --- lb:
    lb = -np.random.uniform(size=(n_x, 1))
    return Q, p, A, b, lb, ub


def generate_random_qp_torch(n_x, m, prob, seeds):
    n_batch = len(seeds)
    Q = np.zeros((n_batch, n_x, n_x))
    p = np.zeros((n_batch, n_x, 1))
    A = np.zeros((n_batch, m, n_x))
    b = np.zeros((n_batch, m, 1))
    lb = np.zeros((n_batch, n_x, 1))
    ub = np.zeros((n_batch, n_x, 1))
    for i in range(n_batch):
        Q_i, p_i, A_i, b_i, lb_i, ub_i = generate_random_qp(n_x=n_x, m=m, prob=prob, seed=seeds[i])
        Q[i, :, :] = Q_i
        p[i, :, :] = p_i
        A[i, :, :] = A_i
        b[i, :, :] = b_i
        lb[i, :, :] = lb_i
        ub[i, :, :] = ub_i

    Q = torch.as_tensor(Q, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    A = torch.as_tensor(A, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    lb = torch.as_tensor(lb, dtype=torch.float64)
    ub = torch.as_tensor(ub, dtype=torch.float64)
    return Q, p, A, b, lb, ub


def generate_random_A(n_x, m, prob):
    # --- A:
    A = np.zeros((m, n_x))
    for i in range(m):
        a = np.random.normal(size=(1, n_x))
        b = np.zeros(1)
        while b.sum() == 0:
            b = np.random.binomial(1, prob, size=(1, n_x))
        a = a*b
        A[i, :] = a

    return A