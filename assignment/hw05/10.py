import numpy as np
import cvxpy as cp

if __name__ == '__main__':
    np.random.seed(2021) # Set random seed so results are repeatable # Set parameters
    M = 100
    N = 1000
    S = 10
    # Define A and y
    A = np.random.randn(M, N)
    ind0 = np.random.choice(N, S, 0) # index subset x0 = np.zeros(N)
    x0 = np.zeros(N)
    x0[ind0] = np.random.rand(S)
    y = A@x0 + .25*np.random.randn(M, 1)

    x = cp.Variable(N)
    tao = 1
    cost = cp.norm(y - A @ x, 2) + tao * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    x_res = x.value

