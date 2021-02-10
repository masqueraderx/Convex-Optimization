'''
Author: Xueren Ge
Time: 2021/1/24
Descrption:
This script models the spread of virus and how
to find the optimal Number of infected individuals
and optimal Number of splits of each sample
'''

import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ## PROBLEM 9
    # Construct the problem
    N = 1000 # Population size
    S = 10 # Number of infected individuals
    M = 100 # Number of tests
    K = 10 # Number of splits of each sample

    # Define x0
    ind0 = np.random.choice(N,S,0) # index subset
    x0 = np.zeros(N)
    x0[ind0] = np.random.rand(S)

    # Define A
    A = np.zeros((M,N))
    for i in np.arange(N):
        ind = np.random.choice(M,K,0)
        A[ind,i] = 1

    y = A @ x0

    x = cp.Variable(N)

    constraints = [x >= 0]

    cost = cp.norm(A @ x - y, 1)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    print("The optimal value is", prob.value)
    print("The optimal x is")
    print(list(round(i, 2) for i in x.value[x.value > 1e-5]))
    print("Predicted number of infected people")
    print(np.argwhere(x.value > 1e-5).tolist())
    print('True number of infected people')
    print(sorted(ind0))
    print("The norm of the residual is ", round(cp.norm(A @ x - y, p=1).value, 2))
    print('===========================================================================\n\n')




    '''
    This part solves (b) in Problem 9
    '''
    print('if we decrease K from 10')
    optimal_k = 10
    for k in range(10, -1, -1):

        ## PROBLEM 9
        # Construct the problem
        N = 1000  # Population size
        S = 10  # Number of infected individuals
        M = 100  # Number of tests

        # Define x0
        ind0 = np.random.choice(N, S, 0)  # index subset
        x0 = np.zeros(N)
        x0[ind0] = np.random.rand(S)

        # Define A
        A = np.zeros((M, N))
        for i in np.arange(N):
            ind = np.random.choice(M, k, 0)
            A[ind, i] = 1

        y = A @ x0
        x = cp.Variable(N)

        constraints = [x >= 0]

        cost = cp.norm(A @ x - y, 1)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        if np.argwhere(x.value > 1e-5).shape[0] != len(ind0):
            break
        else:
            optimal_k = k
    print('the optimal k = ', optimal_k)
    print('\n')
    print('If we increase K from 10')
    optimal_k = 10
    for k in range(11, 101):
        ## PROBLEM 9
        # Construct the problem
        N = 1000  # Population size
        S = 10  # Number of infected individuals
        M = 100  # Number of tests

        # Define x0
        ind0 = np.random.choice(N, S, 0)  # index subset
        x0 = np.zeros(N)
        x0[ind0] = np.random.rand(S)

        # Define A
        A = np.zeros((M, N))
        for i in np.arange(N):
            ind = np.random.choice(M, k, 0)
            A[ind, i] = 1

        y = A @ x0
        x = cp.Variable(N)

        constraints = [x >= 0]

        cost = cp.norm(A @ x - y, 1)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        if np.argwhere(x.value > 1e-5).shape[0] != len(ind0):
            break
        else:
            optimal_k = k
    print('the optimal k = ', optimal_k)
    print('===========================================================================\n\n')


    '''
    This part solves (c) in Problem 9
    '''
    points = []
    for k in range(1, 101):
        temp_max_s = 0
        for s in range(10, 1001):
            ## PROBLEM 9
            # Construct the problem
            N = 1000  # Population size
            M = 100  # Number of tests

            # Define x0
            ind0 = np.random.choice(N, s, 0)  # index subset
            x0 = np.zeros(N)
            x0[ind0] = np.random.rand(s)
            # Define A
            A = np.zeros((M, N))
            for i in np.arange(N):
                ind = np.random.choice(M, k, 0)
                A[ind, i] = 1
            y = A @ x0
            x = cp.Variable(N)

            constraints = [x >= 0]

            cost = cp.norm(A @ x - y, 1)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()
            # if there is a solution
            if np.argwhere(x.value > 1e-5).shape[0] == len(ind0) and temp_max_s < s:
                temp_max_s = s
            else:
                print('current k: {}, current s: {}'.format(k, temp_max_s))
                points.append((k, temp_max_s))
                break

    fid = plt.figure(figsize=(10, 5))
    Axes = plt.subplot(1, 1, 1)
    Axes.axes.tick_params(which='both', direction='in', top=True, right=True)
    plt.minorticks_on()
    Axes.set_facecolor((0, 0, 0, 0.02))
    plt.plot([point[0] for point in points[2:-4]], [point[1] for point in points[2:-4]], 'bs', markersize=3, label='Maximum S')
    plt.plot([point[0] for point in points[:2]], [point[1] for point in points[:2]], 'ro', markersize=3, label='No Solution')
    plt.plot([point[0] for point in points[-3:]], [point[1] for point in points[-3:]], 'ro', markersize=3)
    plt.grid(True, which='major', linewidth=0.5)
    plt.grid(True, which='minor', linewidth=0.1)
    plt.legend()
    plt.title("Relation between $K$ and $S$")
    plt.xlabel("K")
    plt.ylabel("S")
    plt.show()






