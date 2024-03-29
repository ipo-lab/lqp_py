import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
from qpth.qp import QPFunction
from experiments.utils import create_qp_data, plot_profile_bars
import time as time
import matplotlib.pyplot as plt

# --- create problem data
n_x = 1000
n_features = 5
learning_rate = 0.0005
n_batch = 128
n_samples = 2 * n_x
n_sims = 10
tol = 10 ** -3
n_epochs = 100
n_mini_batch = 32
# --- models:
model_names = ['ADMM', 'OptNet']

# --- set models:
control_fp = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol, rho=None, adaptive_rho=True, scale=True)
QP_fp = SolveBoxQP(control=control_fp)

control_optnet = {"eps": tol, "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}
QP_optnet = QPFunction(**control_optnet)

models = {"ADMM": QP_fp, "OptNet": QP_optnet}

# --- storage:
loss_hist = np.zeros((n_sims, n_epochs, len(model_names)))
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)

# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))
    k = -1
    for model_name in model_names:
        # --- define data:
        torch.manual_seed(i)
        k = k + 1
        Q, p, A, b, lb, ub, G, h = create_qp_data(n_x, n_batch, n_samples, seed=i, requires_grad=False)
        QP = models.get(model_name)
        x = torch.normal(mean=0, std=1, size=(n_batch, n_features))
        beta = torch.normal(mean=0, std=1, size=(n_features, n_x))
        p = torch.matmul(x, beta).unsqueeze(2)

        # --- define the model and optimizer:
        m = torch.nn.Linear(n_features, n_x)
        optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate)

        # ---- main training loop
        for epoch in range(n_epochs):

            # --- mini batch index:
            idx = np.random.randint(low=0, high=n_batch, size=n_mini_batch)

            # get output from the model, given the inputs
            p_hat = m(x[idx, :])
            p_hat = p_hat.unsqueeze(2)

            # --- invoke optimization layer:
            if model_name == "OptNet":
                # --- Forward
                start = time.time()
                z = QP(Q[idx, :, :], p_hat[:, :, 0], G[idx, :, :], h[idx, :, 0],  A[idx, :, :], b[idx, :, 0])
                z = z.unsqueeze(2)
                forward_time = time.time() - start
            else:
                # --- Forward
                start = time.time()
                z = QP.forward(Q=Q[idx, :, :], p=p_hat, A=A[idx, :, :], b=b[idx, :, :], lb=lb[idx, :, :],
                               ub=ub[idx, :, :])
                forward_time = time.time() - start
            # --- evaluate losss
            loss = 0.5 * torch.matmul(torch.matmul(torch.transpose(z, 1, 2), Q[idx, :, :]), z).sum() + (
                        p[idx, :, :] * z).sum()

            optimizer.zero_grad()
            # --- compute gradients
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            # --- update parameters
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # --- storage:
            loss_hist[i, epoch, k] = loss.item()
            total_time = forward_time + backward_time
            forward_times[model_name][i] += forward_time
            backward_times[model_name][i] += backward_time
            total_times[model_name][i] += total_time

file_name = 'images_paper/exp_2_dz_'+str(n_x)+'.pdf'
plot_profile_bars(backward_times, forward_times, total_times, n_sims, fontsize=8, logy=True)
plt.savefig(file_name)

# --- convergence plots:
loss_min = loss_hist.min()
loss_dat = loss_hist / abs(loss_min)
loss_mean = pd.concat({
    "ADMM": pd.DataFrame(loss_dat[:, :, 0].mean(axis=0))[0],
    "OptNet": pd.DataFrame(loss_dat[:, :, 1].mean(axis=0))[0]
}, axis=1)
loss_error = pd.concat({
    "ADMM": pd.DataFrame(loss_dat[:, :, 0].std(axis=0))[0],
    "OptNet": pd.DataFrame(loss_dat[:, :, 1].std(axis=0))[0]
}, axis=1)
loss_error = loss_error / n_sims ** 0.5
color2 = ["#0000EE", "#CD3333"]
loss_mean.plot.line(ylabel='Normalized QP Loss', xlabel='Epoch', color=color2, linewidth=4)  # title=title
plt.fill_between(np.arange(n_epochs), loss_mean['ADMM'] - 2 * loss_error['ADMM'],
                 loss_mean['ADMM'] + 2 * loss_error['ADMM'], alpha=0.25, color=color2[0])
plt.fill_between(np.arange(n_epochs), loss_mean['OptNet'] - 2 * loss_error['OptNet'],
                 loss_mean['OptNet'] + 2 * loss_error['OptNet'], alpha=0.25, color=color2[1])
file_name = 'images_paper/exp_2_conv_dz_'+str(n_x)+'.pdf'
plt.savefig(file_name)