import numpy as np
from lqp_py import solve_box_qp_admm as solve_qp
from lqp_py.control import box_qp_control
import time as time

# --- Create problem data
np.random.seed(0)
n_x = 1000
n_samples = 2000
L = np.random.normal(size=(n_samples, n_x))

Q = np.dot(L.T, L)
Q = Q / n_samples
p = np.random.normal(size=(n_x, 1))
A = np.ones((1, n_x))
b = np.ones((1, 1))

lb = -np.ones((n_x, 1))
ub = np.ones((n_x, 1))

# --- Solve time
start = time.time()
sol = solve_qp.solve_box_qp(Q=Q,
                            p=p,
                            A=A,
                            b=b,
                            lb=lb,
                            ub=ub,
                            control=box_qp_control(verbose=True))
end = time.time() - start
print('computation time: {:f}'.format(end))
