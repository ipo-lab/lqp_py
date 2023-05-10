import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.optnet import OptNet
from lqp_py.scs_qp import SolveScsQP
from lqp_py.control import box_qp_control, optnet_control, scs_control
from experiments.utils import create_qp_data
import time as time

# --- create problem data
n_x = 1000
n_batch = 128
n_samples = 2*n_x
n_sims = 10
tol = 10**-5
#tols = [10**-1,10**-3,10**-5]
# --- models:
model_names = ['ADMM_FP','ADMM_Unroll','ADMM_KKT','OptNet','Scs']

# --- set models:
control_fp = box_qp_control(eps_rel=tol, eps_abs=tol, verbose=False, reduce='max')
QP_fp = SolveBoxQP(control=control_fp)

control_unroll = box_qp_control(eps_rel=tol, eps_abs=tol, verbose=False, reduce='max',unroll=True)
QP_unroll = SolveBoxQP(control=control_unroll)

control_kkt = box_qp_control(eps_rel=tol, eps_abs=tol, verbose=False, reduce='max',backward='kkt')
QP_kkt = SolveBoxQP(control=control_unroll)

control_optnet = optnet_control(tol=tol, verbose=False, reduce='max')
QP_optnet = OptNet(control=control_optnet)

control_scs = scs_control(eps_rel=tol, eps_abs=tol)
QP_scs = SolveScsQP(control=control_scs)

models = {"ADMM FP": QP_fp,
          "ADMM Unroll": QP_unroll,
          "ADMM KKT": QP_kkt,
          "OptNet": QP_optnet,
          "Scs": QP_scs}
model_names=list(models.keys())

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
        if model_name == "OptNet" or model_name == "Scs":
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
        dl_dz = torch.ones((n_batch, n_x, 1))
        start = time.time()
        test = x.backward(dl_dz)  # --- slower because updating grads
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
error = 2*error/n_sims**0.5

#title='Box QP Computation Time'
#title='Box QP computation time\n No. Variables: {:d}, Batch size: {:d}, Stopping tolerance: {:.1E}'.format(n_x,n_batch,tol)
#title='No. Variables: {:d}, Batch Size: {:d}, Stopping Tolerance: {:.1E}'.format(n_x,n_batch,tol)
title='Decision Variables = {:d}'.format(n_x)

color = ["#E69F00","#56B4E9","#999999"]
dat.plot.bar(ylabel='time (s)', rot=0, color=color, yerr=error)#title=title,
