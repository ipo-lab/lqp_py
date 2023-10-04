import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.scs_qp import SolveScsQP
from lqp_py.control import box_qp_control, scs_control
from experiments.utils import create_qp_data, plot_profile_bars
from qpth.qp import QPFunction
import time as time
import matplotlib.pyplot as plt

# --- create problem data
n_x = 500
m = 1
n_batch = 128
n_samples = 2 * n_x
n_sims = 10
tol = [1e-1, 1e-3, 1e-5]

# --- set models:
QP_fp = []
QP_unroll = []
QP_kkt = []
QP_optnet = []
QP_scs = []
for i in range(len(tol)):
    QP_fp.append(SolveBoxQP(control=box_qp_control(eps_rel=tol[i], eps_abs=tol[i])))
    QP_kkt.append(SolveBoxQP(control=box_qp_control(eps_rel=tol[i], eps_abs=tol[i], backward='kkt')))
    QP_unroll.append(SolveBoxQP(control=box_qp_control(eps_rel=tol[i], eps_abs=tol[i], unroll=True)))
    QP_optnet.append(
        QPFunction(**{"eps": tol[i], "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}))
    QP_scs.append(SolveScsQP(control=scs_control(eps_rel=tol[i], eps_abs=tol[i])))

models = {"ADMM FP 1": QP_fp[0], "ADMM FP 3": QP_fp[1], "ADMM FP 5": QP_fp[2],
          "ADMM KKT 1": QP_kkt[0], "ADMM KKT 3": QP_kkt[1], "ADMM KKT 5": QP_kkt[2],
          "ADMM Unroll 1": QP_unroll[0], "ADMM Unroll 3": QP_unroll[1], "ADMM Unroll 5": QP_unroll[2],
          "OptNet 1": QP_optnet[0], "OptNet 3": QP_optnet[1], "OptNet 5": QP_optnet[2],
          "SCS 1": QP_scs[0], "SCS 3": QP_scs[1], "SCS 5": QP_scs[2]}

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

    for model_name in model_names:
        print("model: {:s}".format(model_name))
        Q, p, A, b, lb, ub, G, h = create_qp_data(n_x, n_batch, n_samples, seed=i)
        QP = models.get(model_name)
        # --- forward:
        if "OptNet" in model_name:
            p = p[:, :, 0]
            h = h[:, :, 0]
            b = b[:, :, 0]
            dl_dz = torch.ones((n_batch, n_x))
        else:
            dl_dz = torch.ones((n_batch, n_x, 1))

        if "OptNet" in model_name:
            # --- Forward
            start = time.time()
            x = QP(Q, p, G, h, A, b)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        elif "SCS" in model_name:
            # --- Forward
            start = time.time()
            x = QP.forward(Q=Q, p=p, A=A, b=b, G=G, h=h)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        else:
            # --- Forward
            start = time.time()
            x = QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))

        # --- backward
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

file_name = 'images_paper/dz_'+str(n_x)+'.pdf'
plot_profile_bars(backward_times, forward_times, total_times, n_sims, fontsize=8, logy=True, figsize=(18, 6))
plt.savefig(file_name)
