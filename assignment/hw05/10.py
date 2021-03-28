import numpy as np
import cvxpy as cp
import random
import os
from matplotlib import pyplot as plt

def Q10_a():
    res = []
    for tau in np.arange(0.01, 10, 0.01):
        x = cp.Variable(N)
        cost = 0.5 * cp.sum_squares(y - A @ x) + tau * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        res.append(cp.norm(x.value - x0, p=2).value)
    optimal_tau = (res.index(min(res)) + 1) / 100
    print('the optimal value is {}'.format(optimal_tau))
    return res, optimal_tau

def funcValue(tau, A, x, y):
    var = y - A @ x
    return 0.5 * np.linalg.norm(x=var, ord=2) ** 2 + tau * np.linalg.norm(x=x, ord=1)

def subGradient(alpha, tau, A, x, y, maxiter, delta, flag):
    k = 1
    x = np.mat(x).T #(N ,1)
    y = np.mat(y).T #(M, 1)
    A = np.mat(A) #(M, N)
    value = []
    iteration = []
    while (k < maxiter) and (np.linalg.norm(x - np.mat(x0)) > delta):
        dk = A.T @ (A @ x - y) + tau * np.sign(x) #(N, 1)
        if flag:
            x = x - alpha * dk / np.sqrt(k) # (N, 1)
        else:
            x = x - alpha * dk / k
        value.append(funcValue(tau, A, x, y))
        iteration.append(k)
        k += 1
        print(x.flatten())
    return x, value, iteration


def Proximal(alpha, tau, A, x, y, maxiter, delta):
    def T(x):
        for i in range(len(x)):
            if x[i] >= tau * alpha:
                x[i] = x[i] - tau * alpha
            elif abs(x[i]) <= tau * alpha:
                x[i] = 0
            elif x[i] <= - tau * alpha:
                x[i] = x[i] + tau * alpha
            else:
                return -1
        return x
    k = 0
    x = np.mat(x).T #(N ,1)
    y = np.mat(y).T #(M, 1)
    A = np.mat(A) #(M, N)
    fValue = []
    iteration = []
    while (k < maxiter) and (np.linalg.norm(x - np.mat(x0)) > delta):
        x = T(x + alpha * A.T @ (y - A @ x))
        k += 1
        print(x.flatten())
        fValue.append(funcValue(tau, A, x, y))
        iteration.append(k)
    return x, fValue, iteration


def accrProximal(alpha, tau, A, x, y, maxiter, delta):
    def Tec(x):
        for i in range(len(x)):
            if x[i] >= tau * alpha:
                x[i] = x[i] - tau * alpha
            elif abs(x[i]) <= tau * alpha:
                x[i] = 0
            elif x[i] <= - tau * alpha:
                x[i] = x[i] + tau * alpha
            else:
                return -1
        return x
    k = 1
    pk = np.mat(np.zeros(N)).T #(N, 1)

    x = [np.mat(x).T] #(N ,1)
    y = np.mat(y).T #(M, 1)
    A = np.mat(A) #(M, N)

    gd = A.T @ (A @ (x[-1] + pk) - y) #(N, 1)
    fValue = []
    iteration = []
    while (k < maxiter) and (np.linalg.norm(x[-1] - np.mat(x0)) > delta):
        x.append(Tec(x[-1] + pk - alpha * gd))
        k += 1
        pk = ((k - 1) / (k + 2)) * (x[-1] - x[-2])
        gd = A.T @ (A @ (x[-1] + pk) - y)

        print(x[-1].flatten())
        fValue.append(funcValue(tau, A, x[-1], y))
        iteration.append(k)
    return x[-1], fValue, iteration

if __name__ == '__main__':
    np.random.seed(2021) # Set random seed so results are repeatable # Set parameters
    M = 100
    N = 1000
    S = 10
    # Define A and y
    A = np.random.randn(M, N) # (M, N)
    ind0 = np.random.choice(N, S, 0)
    x0 = np.zeros(N)
    x0[ind0] = np.random.rand(S) # (N, )
    y = A@x0 + .25*np.random.randn(M) # (M, )
    savepath = 'C:/gexueren/桌面/hw05/hw05/hw05'

############################################# Q1 #####################################################
    res, optimal_tau = Q10_a()
    plt.figure(figsize=(10, 5))

    plt.plot(np.arange(0.01, 10, 0.01), res, 'r-*', label='tau')
    plt.title('Best Tau is {}'.format(optimal_tau))
    plt.xlabel('tau')
    plt.ylabel('norm residual')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'Q10_1.jpg'))

############################################# Q2 #####################################################
    x = np.ones(N) # 初值 (N, )
    _, fValue_0, iteration_0 = subGradient(alpha=0.001, tau=optimal_tau, A=A, x=x, y=y, maxiter=5000, delta=1e-3, flag=True)
    plt.figure(figsize=(10, 5))
    plt.plot(iteration_0, fValue_0, color='r', label=r'$\alpha_{k}= \frac{\alpha}{\sqrt{k}}$')
    plt.title('SubGradient Algo')
    plt.xlabel('iterations')
    plt.ylabel('Value of Objective function')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'Q10_2_1.jpg'))

    x = np.ones(N) # 初值 (N, )
    _, fValue_1, iteration_1 = subGradient(alpha=0.03, tau=optimal_tau, A=A, x=x, y=y, maxiter=5000, delta=1e-3, flag=False)
    plt.plot(iteration_1, fValue_1, color='g', label=r'$\alpha_{k}= \frac{\alpha}{k}$')
    plt.title('SubGradient Algo')
    plt.xlabel('iterations')
    plt.ylabel('Value of Objective function')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'Q10_2_2.jpg'))

# ############################################# Q3 #####################################################
    x = np.ones(N) # 初值 (N, )
    _, fValue, iteration = Proximal(alpha=0.001, tau=optimal_tau, A=A, x=x, y=y, maxiter=1000, delta=1e-3)
    plt.figure(figsize=(10, 5))
    plt.plot(iteration, fValue)
    plt.xlabel('iterations')
    plt.ylabel('Value of Objective function')
    plt.title('Proximal Gradient Algo')
    plt.savefig(os.path.join(savepath, 'Q10_3.jpg'))
#
# ############################################# Q4 #####################################################
    x = np.ones(N) # 初值 (N, )
    _, fValue, iteration = accrProximal(alpha=0.0001, tau=optimal_tau, A=A, x=x, y=y, maxiter=1000, delta=1e-3)
    plt.figure(figsize=(10, 5))
    plt.plot(iteration, fValue)
    plt.xlabel('iterations')
    plt.ylabel('Value of Objective function')
    plt.title('Proximal Accelerated Gradient Algo')
    plt.savefig(os.path.join(savepath, 'Q10_4.jpg'))