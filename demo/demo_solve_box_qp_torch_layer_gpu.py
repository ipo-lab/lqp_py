import torch
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
import time as time

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- create problem data
torch.manual_seed(0)
n_x = 1000
n_batch = 128
n_samples = 2 * n_x
tol = 10**-6
L = torch.randn(n_batch, n_samples, n_x).to(device)

Q = torch.matmul(torch.transpose(L, 1, 2), L)
Q = Q / n_samples
Q.requires_grad = True
p = torch.randn(n_batch, n_x, 1, requires_grad=True).to(device)
A = torch.ones(n_batch, 1, n_x, requires_grad=True).to(device)
b = torch.ones(n_batch, 1, 1, requires_grad=True).to(device)

lb = -torch.ones(n_batch, n_x, 1, requires_grad=True).to(device)
ub = torch.ones(n_batch, n_x, 1, requires_grad=True).to(device)

# --- QP solver
control = box_qp_control(eps_rel=tol, eps_abs=tol, verbose=True, reduce="max")
QP = SolveBoxQP(control=control).to(device)

# --- Forward
start = time.time()
x = QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
end = time.time() - start
print("computation time: {:f}".format(end))

# --- Backward
dl_dz = torch.ones((n_batch, n_x, 1), device=device)

start = time.time()
test = x.backward(dl_dz)
end = time.time() - start
print("computation time: {:f}".format(end))


# --- QP solver: KKT
control = box_qp_control(
    eps_rel=tol, eps_abs=tol, verbose=True, reduce="max", backward="kkt"
)
QP = SolveBoxQP(control=control).to(device)

# --- Forward
start = time.time()
x = QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
end = time.time() - start
print("computation time: {:f}".format(end))

# --- Backward
dl_dz = torch.ones((n_batch, n_x, 1), device=device)

start = time.time()
test = x.backward(dl_dz)
end = time.time() - start
print("computation time: {:f}".format(end))


# --- QP solver: unroll
control = box_qp_control(
    eps_rel=tol, eps_abs=tol, verbose=True, reduce="max", unroll=True
)
QP = SolveBoxQP(control=control).to(device)

# --- Forward
start = time.time()
x = QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
end = time.time() - start
print("computation time: {:f}".format(end))

# --- Backward
dl_dz = torch.ones((n_batch, n_x, 1), device=device)

start = time.time()
test = x.backward(dl_dz)
end = time.time() - start
print("computation time: {:f}".format(end))
