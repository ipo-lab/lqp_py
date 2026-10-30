import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
from qpth.qp import QPFunction
from experiments.utils import load_data, plot_profile_bars
import time as time
import matplotlib.pyplot as plt

# --- Experiment params
learning_rate = 0.01
n_batch = 32
n_sims = 10
tol = 1e-3
n_epochs = 500
dtype = torch.float64
scale_factor = 250
date_is_end = "2014"
date_oos_start = "2015"
lookback = 52
annual_factor = 52 ** 0.5

# --- load data and partition is/oos
x0, y0 = load_data()
x0 = x0 * scale_factor
y0 = y0 * scale_factor


# --- Expected Return features: Simple moving average
x0 = x0.rolling(lookback, min_periods=1).mean() * np.sqrt(lookback)
x0 = x0.shift(1).dropna()

y0 = y0[x0.index[0]:]

# --- in sample and out of sample
x_is_df = x0[:date_is_end]
y_is_df = y0[:date_is_end]
x_oos_df = x0[date_oos_start:]
y_oos_df = y0[date_oos_start:]

n = y_is_df.shape[1]
n_is = x_is_df.shape[0]
n_oos = x_oos_df.shape[0]

# --- Rolling covariance:
y0_shift = y0.shift(1)
var = y0_shift.ewm(span=lookback, min_periods=2).var().bfill()# backfill only affects training
covar = y0_shift.ewm(span=lookback).cov()
covar = covar.fillna(0.0)
covar = np.reshape(covar.values, [y0.shape[0], y0.shape[1], y0.shape[1]])
D = np.zeros([y0.shape[0], y0.shape[1], y0.shape[1]])
D[:, np.arange(n), np.arange(n)] = var.values
covar_final = 0.5 * covar + 0.5 * D
covar_is = covar_final[:n_is]
covar_oos = covar_final[-n_oos:]

# --- remove initialization period:
x_is_df=x_is_df.iloc[lookback:]
y_is_df=y_is_df.iloc[lookback:]
covar_is = covar_is[lookback:]
n_is = x_is_df.shape[0]

# --- Create OLS benchmark portfolios:
beta_ols = np.linalg.lstsq(x_is_df, y_is_df, rcond=None)[0]
y_hat_ols_is = x_is_df.values.dot(beta_ols)
y_hat_ols_oos = x_oos_df.values.dot(beta_ols)


# --- Compute the covariance matrix -- simple in-sample estimate
Q_is = torch.tensor(covar_is, dtype=dtype)
Q_oos = torch.tensor(covar_oos, dtype=dtype)
p_is = torch.zeros(size=[n_batch, n, 1], dtype=dtype)
p_oos = torch.zeros(size=[n_oos, n, 1], dtype=dtype)

# Constraint set:
# A is dynamic:
b_is = torch.ones(size=[n_batch, 1, 1], dtype=dtype)
b_oos = torch.ones(size=[n_oos, 1, 1],dtype=dtype)

lb_is = torch.zeros(size=[n_batch, 1, 1 ],dtype=dtype)
lb_oos = torch.zeros(size=[n_oos, 1, 1 ], dtype=dtype)
ub_is = 1000. * torch.ones(size=[n_batch, 1, 1],dtype=dtype)
ub_oos = 1000. * torch.ones(size=[n_oos, 1, 1],dtype=dtype)
G = -torch.eye(n,dtype=dtype)
G_is = G.unsqueeze(0).repeat(n_batch, 1, 1)
G_oos = G.unsqueeze(0).repeat(n_oos, 1, 1)

# Variables as tensor:
x_is = torch.tensor(x_is_df.values, dtype=dtype)
x_oos = torch.tensor(x_oos_df.values, dtype=dtype)
y_is = torch.tensor(y_is_df.values, dtype=dtype)
y_oos = torch.tensor(y_oos_df.values, dtype=dtype)

# --- models:
model_names = ['ADMM', 'OptNet']

# --- set models:
control_fp = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol, rho=None, adaptive_rho=False, scale=True,
                            verbose=False)
QP_fp = SolveBoxQP(control=control_fp)

control_oos = box_qp_control(max_iters=10_000, eps_rel=tol/100., eps_abs=tol/100., rho=None, adaptive_rho=False, scale=True,
                            verbose=False)
QP_oos = SolveBoxQP(control=control_oos)

control_optnet = {"eps": tol, "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}
QP_optnet = QPFunction(**control_optnet)

models = {"ADMM": QP_fp, "OptNet": QP_optnet, "OOS": QP_oos}

# --- storage:
tuples = [("ADMM", i) for i in range(n_sims)] + [("OptNet", i) for i in range(n_sims)]
columns = pd.MultiIndex.from_tuples(tuples)
loss_hist = pd.DataFrame(0., index=range(n_epochs),columns=columns)
rets_oos = pd.DataFrame(0., index=x_oos_df.index,columns=columns)
forward_times = pd.DataFrame(np.zeros((n_sims, len(model_names))), columns=model_names)
backward_times = pd.DataFrame(np.zeros((n_sims, len(model_names))), columns=model_names)
total_times = pd.DataFrame(np.zeros((n_sims, len(model_names))), columns=model_names)


# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))
    for model_name in model_names:
        # --- define data:
        torch.manual_seed(i)
        np.random.seed(i)
        QP = models.get(model_name)

        # --- define the model and optimizer:
        m = torch.nn.Linear(x_is.shape[1], y_is.shape[1], bias=False, dtype=dtype)
        m.weight.data = torch.tensor(beta_ols.T)
        optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

        # ---- main training loop
        for epoch in range(n_epochs):
            # --- mini batch index:
            idx = np.random.randint(low=0, high=n_is, size=n_batch)

            # get output from the model, given the inputs
            p_hat = m(x_is[idx, :])
            p_hat = p_hat.unsqueeze(2)
            A = torch.transpose(p_hat, 2, 1)
            Q = Q_is[idx]

            # --- invoke optimization layer:
            if model_name == "OptNet":
                # --- Forward
                start = time.time()
                z = QP(Q, p_is.squeeze(-1), G_is, lb_is.squeeze(-1), A, b_is.squeeze(-1))
                forward_time = time.time() - start
            else:
                # --- Forward
                start = time.time()
                z = QP.forward(Q=Q, p=p_is, A=A, b=b_is, lb=lb_is, ub=ub_is)
                forward_time = time.time() - start
                z = z.squeeze(-1)
            # --- evaluate losss
            z = z/z.sum(dim=1).unsqueeze(1)
            r = (z * y_is[idx]).sum(dim=1)

            loss = -r.mean()/r.std()

            optimizer.zero_grad()
            # --- compute gradients
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            # --- update parameters
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # --- storage:
            loss_hist.loc[epoch, (model_name,i)] = loss.item()
            total_time = forward_time + backward_time
            forward_times[model_name][i] += forward_time
            backward_times[model_name][i] += backward_time
            total_times[model_name][i] += total_time

        # --- out of sample results:
        p_hat = m(x_oos)
        p_hat = p_hat.unsqueeze(2)
        A = torch.transpose(p_hat, 2, 1)
        QP = models.get("OOS")
        z = QP.forward(Q=Q_oos, p=p_oos, A=A, b=b_oos, lb=lb_oos, ub=ub_oos)
        z = z.squeeze(-1)
        z = torch.clamp_min(z, 0.0)
        z = z / z.sum(dim=1).unsqueeze(1)
        r = (z * y_oos).sum(dim=1)
        r = pd.Series(r.detach().numpy(), index=x_oos_df.index)
        print(r.mean()/r.std() * annual_factor)
        rets_oos.loc[:,(model_name, i)] = r




# --- Make the plots:
file_name = 'images_paper/exp_3_runtime.pdf'
plot_profile_bars(backward_times, forward_times, total_times, n_sims, fontsize=8, logy=True)
plt.savefig(file_name)

# --- convergence plots:
loss_sma = loss_hist.rolling(10,min_periods=1).mean()
loss_mean = loss_sma.groupby(axis=1, level=0).mean()
loss_error = loss_sma.groupby(axis=1, level=0).std()
loss_error = loss_error / n_sims ** 0.5
color2 = ["#0000EE", "#CD3333"]
loss_mean.plot.line(ylabel='Sharpe Ratio Loss', xlabel='Epoch', color=color2, linewidth=4)  # title=title
plt.fill_between(np.arange(n_epochs), loss_mean['ADMM'] - 2 * loss_error['ADMM'],
                 loss_mean['ADMM'] + 2 * loss_error['ADMM'], alpha=0.25, color=color2[0])
plt.fill_between(np.arange(n_epochs), loss_mean['OptNet'] - 2 * loss_error['OptNet'],
                 loss_mean['OptNet'] + 2 * loss_error['OptNet'], alpha=0.25, color=color2[1])
file_name = 'images_paper/exp_3_conv.pdf'
plt.savefig(file_name)

# --- results oos
# --- get the equal weight benchmark
r_ew = y_oos_df.mean(axis=1)

# --- get the OLS benchmark
A = torch.tensor(y_hat_ols_oos).unsqueeze(1)
QP = models.get("OOS")
z = QP.forward(Q=Q_oos, p=p_oos, A=A, b=b_oos, lb=lb_oos, ub=ub_oos)
z = z.squeeze(-1)
z = torch.clamp_min(z, 0.0)
z = z / z.sum(dim=1).unsqueeze(1)
r = (z * y_oos).sum(dim=1)
r_ols = pd.Series(r.detach().numpy(), index=x_oos_df.index)

# --- IPO results
r_admm=rets_oos.filter(like='ADMM', axis=1).mean(axis=1)
r_optnet=rets_oos.filter(like='OptNet', axis=1).mean(axis=1)
results_oos = pd.concat([r_ew, r_ols, r_admm, r_optnet],axis=1)
results_oos.columns = ['Equal Weight', "OLS", "ADMM", "OptNet"]
results_oos = results_oos/scale_factor
sr = results_oos.mean()/results_oos.std()*annual_factor
results_oos.columns = [f"{col} (Sharpe={round(float(sr[col]),3)})" for col in results_oos.columns]
((1 + results_oos).cumprod() - 1.).plot(figsize=(10,5))
file_name = 'images_paper/exp_3_eq.pdf'
plt.savefig(file_name)