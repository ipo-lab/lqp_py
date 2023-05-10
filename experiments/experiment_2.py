import torch
import numpy as np
import pandas as pd
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.optnet import OptNet
from lqp_py.control import box_qp_control, optnet_control
from experiments.utils import create_qp_data
import time as time
import matplotlib.pyplot as plt

# --- create problem data
n_x = 500
n_features = 5
learning_rate = 0.001
n_batch = 128
n_samples = 2*n_x
n_sims = 10
tol = 10**-5
n_epochs = 30
n_mini_batch = 32
# --- models:
model_names = ['ADMM_FP','OptNet']

# --- set models:
control_fp = box_qp_control(eps_rel=tol, eps_abs=tol, verbose=False, reduce='max')
QP_fp = SolveBoxQP(control=control_fp)

control_optnet = optnet_control(tol=tol, verbose=False, reduce='max')
QP_optnet = OptNet(control=control_optnet)

models = {"ADMM_FP": QP_fp, "OptNet": QP_optnet}

# --- storage:
loss_hist = np.zeros((n_sims,n_epochs,len(model_names)))
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)

# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))
    k=-1
    for model_name in model_names:
        # --- define data:
        torch.manual_seed(i)
        k = k+1
        Q, p, A, b, lb, ub, G, h = create_qp_data(n_x, n_batch, n_samples, seed=i,requires_grad=False)
        QP = models.get(model_name)
        x = torch.normal(mean=0,std=1,size=(n_batch,n_features))
        beta = torch.normal(mean=0,std=1,size=(n_features,n_x))
        p = torch.matmul(x,beta).unsqueeze(2)

        # --- define the model and optimizer:
        m = torch.nn.Linear(n_features, n_x)
        optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate)


        # ---- main training loop
        for epoch in range(n_epochs):

            # --- mini batch index:
            idx = np.random.randint(low=0, high=n_batch, size=n_mini_batch)

            # get output from the model, given the inputs
            p_hat = m(x[idx,:])
            p_hat = p_hat.unsqueeze(2)

            # --- invoke optimization layer:
            if model_name == "OptNet" or model_name == "Scs":
                # --- Forward
                start = time.time()
                z = QP.forward(Q=Q[idx,:,:], p=p_hat, A=A[idx,:,:], b=b[idx,:,:], G=G[idx,:,:], h=h[idx,:,:])
                forward_time = time.time() - start
            else:
                # --- Forward
                start = time.time()
                z = QP.forward(Q=Q[idx,:,:], p=p_hat, A=A[idx,:,:], b=b[idx,:,:], lb=lb[idx,:,:], ub=ub[idx,:,:])
                forward_time = time.time() - start
            # --- evaluate losss
            loss = 0.5*torch.matmul(torch.matmul(torch.transpose(z,1,2),Q[idx,:,:]),z).sum() + (p[idx,:,:]*z).sum()

            optimizer.zero_grad()
            # --- compute gradients
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            # --- update parameters
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # --- storage:
            loss_hist[i,epoch,k] = loss.item()
            total_time = forward_time + backward_time
            forward_times[model_name][i] += forward_time
            backward_times[model_name][i] += backward_time
            total_times[model_name][i] += total_time





# --- median runtime
dat = pd.concat({
    'Backward': backward_times.median(axis=0),
    'Forward': forward_times.median(axis=0),
    "Total": total_times.median(axis=0)}, axis=1)

# --- error bars:
error = pd.concat({
    'Backward': backward_times.std(axis=0),
    'Forward': forward_times.std(axis=0),
    "Total": total_times.std(axis=0)}, axis=1)

title='Decision Variables = {:d}'.format(n_x)

color = ["#E69F00","#56B4E9","#999999"]
dat.plot.bar(ylabel='time (s)', rot=0, color=color, yerr=error)#title=title

# --- convergence plots:
loss_min = loss_hist.min()
loss_dat = loss_hist/abs(loss_min)
loss_mean = pd.concat({
    "ADMM":pd.DataFrame(loss_dat[:,:,0].mean(axis=0))[0],
    "OptNet":pd.DataFrame(loss_dat[:,:,1].mean(axis=0))[0]
},axis=1)
loss_error = pd.concat({
    "ADMM":pd.DataFrame(loss_dat[:,:,0].std(axis=0))[0],
    "OptNet":pd.DataFrame(loss_dat[:,:,1].std(axis=0))[0]
},axis=1)
loss_error = loss_error/n_sims**0.5
color2 = ["#0000EE","#CD3333"]
loss_mean.plot.line(ylabel='Normalized QP Loss',xlabel='Epoch',color=color2,linewidth=4)#title=title
plt.fill_between(np.arange(n_epochs),loss_mean['ADMM']-2*loss_error['ADMM'],
                 loss_mean['ADMM']+2*loss_error['ADMM'],alpha=0.25,color=color2[0])
plt.fill_between(np.arange(n_epochs),loss_mean['OptNet']-2*loss_error['OptNet'],
                 loss_mean['OptNet']+2*loss_error['OptNet'],alpha=0.25,color=color2[1])


#title='Box QP Computation Time'
#title='Box QP computation time\n Number Variables: {:d}, Batch size: {:d},\n Number of epochs: {:d},  Stopping tolerance: {:f}'.format(n_x,n_mini_batch, n_epochs,tol)
#dat.plot.bar(ylabel='time (s)', rot=0, title = title)

plt.plot(loss_hist[:,:,0].mean(axis=0), '-')
plt.plot(loss_hist[:,:,1].mean(axis=0), '-')