import numpy as np
from lqp_py import solve_box_qp_admm as solve_qp
from lqp_py.control import box_qp_control
import time as time

# --- Create problem data
np.random.seed(0)
n_x = 3
n_samples = 10
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


np.random.seed(0)
n_x = 3
n_samples = 200
Q = np.dot(L.T, L)
Q = Q / n_samples
p = np.random.normal(size=(n_x, 1))
A = np.ones((1, n_x))
b = np.ones((1, 1))

lb = np.zeros((n_x, 1))
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

sol.get('z')
0.5*sol.get('z').dot(Q).dot(sol.get('z')) + p.T.dot(sol.get('z'))


import cvxpy as cp

w = cp.Variable(n_x)
constraints = [cp.sum(w) == 1,w>=0]


# Objective Function
obj = 0.5*cp.quad_form(w, Q) + p.T @ w
prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
prob.solve()
prob.solution
sol.get('z')