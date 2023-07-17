import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
from experiments.utils import generate_hard_qp_torch
from qpth.qp import QPFunction
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import time as time

# --- create problem data
n_x = 250
m = round(n_x ** 0.5)
prob = 0.85
n_batch = 128
n_samples = 2 * n_x
n_sims = 10
tol = 10 ** -5

# --- set models:
control_fp = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol, rho=None, adaptive_rho=True, scale=True)
QP_fp = SolveBoxQP(control=control_fp)

control_unroll = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol, rho=None, adaptive_rho=True, scale=True,
                                unroll=True)
QP_unroll = SolveBoxQP(control=control_unroll)

control_kkt = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol, rho=None, adaptive_rho=True, scale=True,
                             backward='kkt')
QP_kkt = SolveBoxQP(control=control_unroll)

control_optnet = {"eps": tol, "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}
QP_optnet = QPFunction(**control_optnet)

# --- cvx:
Q_sqrt = cp.Parameter((n_x, n_x))
p_cvx = cp.Parameter(n_x)
A_cvx = cp.Parameter((m, n_x))
b_cvx = cp.Parameter(m)
G_cvx = cp.Parameter((2 * n_x, n_x))
h_cvx = cp.Parameter(2 * n_x)
x_cvx = cp.Variable(n_x)
z_cvx = cp.Variable(1)
obj = cp.Minimize(0.5 * z_cvx + p_cvx.T @ x_cvx)
cons = [A_cvx @ x_cvx == b_cvx, G_cvx @ x_cvx <= h_cvx, cp.sum_squares(Q_sqrt @ x_cvx) <= z_cvx]
problem = cp.Problem(obj, cons)

QP_scs = CvxpyLayer(problem, parameters=[Q_sqrt, p_cvx, A_cvx, b_cvx, G_cvx, h_cvx], variables=[x_cvx, z_cvx])  # gp=False
control_scs = {'max_iters': 10_000, "eps": tol}

models = {"ADMM FP": QP_fp,
          #"ADMM Unroll": QP_unroll,
          #"ADMM KKT": QP_kkt,
          "OptNet": QP_optnet,
          "Cvxpylayers": QP_scs}
model_names = list(models.keys())

# --- storage:
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)

# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))
    seeds = range(i*n_batch, i*n_batch+n_batch)
    for model_name in model_names:
        print("model: {:s}".format(model_name))
        Q, p, A, b, lb, ub, G, h = generate_hard_qp_torch(n_x, prob, seeds)
        QP = models.get(model_name)
        # --- forward:
        if model_name == "OptNet":
            p = p[:, :, 0]
            h = h[:, :, 0]
            b = b[:, :, 0]
            dl_dz = torch.ones((n_batch, n_x))
        else:
            dl_dz = torch.ones((n_batch, n_x, 1))

        if model_name == "OptNet":
            # --- Forward
            start = time.time()
            x = QP(Q, p, G, h, A, b)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        elif model_name == "Cvxpylayers":
            # --- Forward
            start = time.time()
            x = QP_scs(torch.linalg.cholesky(Q, upper=True), p[:, :, 0], A, b[:, :, 0], G, h[:, :, 0],
                       solver_args=control_scs)
            x = x[0]
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        else:
            # --- Forward
            start = time.time()
            x = QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))

        # --- backward
        if model_name == "Cvxpylayers":
            start = time.time()
            x.sum().backward()
            backward_time = time.time() - start
            print('backward time: {:f}'.format(backward_time))
        else:
            start = time.time()
            test = x.backward(dl_dz)
            backward_time = time.time() - start
            print('backward time: {:f}'.format(backward_time))

        # --- total time:
        total_time = forward_time + backward_time
        print('total time: {:f}'.format(total_time))

        # --- storage:
        forward_times[model_name][i] = forward_time
        backward_times[model_name][i] = backward_time
        total_times[model_name][i] = total_time

# --- median values
dat = pd.concat({
    'Backward': backward_times.median(axis=0),
    'Forward': forward_times.median(axis=0),
    "Total": total_times.median(axis=0)}, axis=1)

# --- error bars:
error = pd.concat({
    'Backward': backward_times.std(axis=0),
    'Forward': forward_times.std(axis=0),
    "Total": total_times.std(axis=0)}, axis=1)
error = 2 * error / n_sims ** 0.5

color = ["#E69F00", "#56B4E9", "#999999"]
dat.plot.bar(ylabel='time (s)', rot=0, color=color, yerr=error)
