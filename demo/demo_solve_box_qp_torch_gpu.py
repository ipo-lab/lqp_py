import torch
from lqp_py.solve_box_qp_admm_torch import torch_solve_box_qp, torch_solve_box_qp_grad
from lqp_py.control import box_qp_control
import time as time

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- create problem data
torch.manual_seed(0)
n_x = 500
n_batch = 128
n_samples = 2 * n_x
L = torch.randn(n_batch, n_samples, n_x)

Q = torch.matmul(torch.transpose(L, 1, 2), L)
Q = Q / n_samples
p = torch.randn(n_batch, n_x, 1).to(device)
A = torch.ones(n_batch, 1, n_x).to(device)
b = torch.ones(n_batch, 1, 1).to(device)

lb = -torch.ones(n_batch, n_x, 1).to(device)
ub = torch.ones(n_batch, n_x, 1).to(device)

# --- Forward
control = box_qp_control(eps_rel=10**-6, eps_abs=10**-6, verbose=True, reduce="max")
start = time.time()
sol = torch_solve_box_qp(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub, control=control)
end = time.time() - start
print("computation time: {:f}".format(end))

# --- Backward
dl_dz = torch.ones((n_batch, n_x, 1)).to(device)
# dl_dz = torch.randn((n_batch,n_x,1))
start = time.time()
grads = torch_solve_box_qp_grad(
    dl_dz=dl_dz,
    x=sol.get("x"),
    u=sol.get("u"),
    lams=sol.get("lams"),
    nus=sol.get("nus"),
    Q=Q,
    A=A,
    lb=lb,
    ub=ub,
    rho=control.get("rho"),
)
end = time.time() - start
print("computation time: {:f}".format(end))
