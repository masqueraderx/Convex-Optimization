'''
Author: Xueren Ge
Time: 2021/1/23
Descrption:
This script find the optimal solution for different norm
'''
import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp

if __name__ == '__main__':
    # Define function
    f_opt = lambda t: np.cos(2 * np.pi * t)

    # Number of samples
    M = 100

    # Degree + 1
    N = 9

    # Choose sample location
    tt = np.linspace(-1, 1, M)

    # Compute "clean" function
    ff = f_opt(tt)

    # Form "observation matrix"
    A = np.zeros((M, N))
    for i in np.arange(N):
        A[:, i] = tt ** i

    # Compute and plot least squares polynomial fit for "noise free" samples
    xhat = np.linalg.pinv(A) @ ff
    fhat = A @ xhat


    ## Construct noisy observations
    np.random.seed(2020)  # Set random seed so results are repeatable

    # Random gaussian noise
    noise1 = np.random.randn(M)
    noise1 = 5 * noise1 / np.linalg.norm(noise1)
    y1 = ff + noise1
    xhat_1 = np.linalg.pinv(A) @ y1
    yhat_1 = A @ xhat_1
    print('Random gaussian noise: ', list(round(i, 2) for i in xhat_1))
    print("The norm of the residual is ", round(cp.norm(A @ xhat_1 - ff, p=2).value, 2))
    print('\n')

    # Sparse gaussian noise
    K = 15
    Gamma = np.random.choice(M, K, 0)
    noise2 = np.zeros(M)
    noise2[Gamma] = np.random.randn(K)
    noise2 = 5 * noise2 / np.linalg.norm(noise2)
    y2 = ff + noise2
    xhat_2 = np.linalg.pinv(A) @ y2
    yhat_2 = A @ xhat_2
    print('Sparse gaussian noise: ', list(round(i, 2) for i in xhat_2))
    print("The norm of the residual is ", round(cp.norm(A @ xhat_2 - ff, p=2).value, 2))
    print('\n')

    # Uniform noise
    noise3 = np.random.rand(M) - 0.5
    noise3 = 5 * noise3 / np.linalg.norm(noise3)
    y3 = ff + noise3
    xhat_3 = np.linalg.pinv(A) @ y3
    yhat_3 = A @ xhat_3
    print('Uniform noise: ', list(round(i, 2) for i in xhat_3))
    print("The norm of the residual is ", round(cp.norm(A @ xhat_3 - ff, p=2).value, 2))
    print('\n')


    fid = plt.figure(figsize=(10, 5))
    Axes = plt.subplot(1, 1, 1)
    Axes.axes.tick_params(which='both', direction='in', top=True, right=True)
    plt.minorticks_on()
    Axes.set_facecolor((0, 0, 0, 0.02))
    plt.plot(tt, ff, color='k', label='Origin f(t)')
    plt.plot(tt, fhat, 'k--', color='r', label='Polynomial fit')
    plt.plot(tt, yhat_1, 'k--', color='g', label='Random gaussian noise')
    plt.plot(tt, yhat_2, 'k--', color='b', label='Sparse gaussian noise')
    plt.plot(tt, yhat_3, 'k--', color='m', label='Uniform noise')
    plt.grid(True, which='major', linewidth=0.5)
    plt.grid(True, which='minor', linewidth=0.1)
    plt.xlabel('t')
    plt.title('Polynomial interpolation of f(t), Degree = {}, norm = {}'.format(N - 1, 2))
    plt.legend(loc='upper right', fontsize='x-small')
    plt.show()



    '''
    This part solves the norm infinity and norm 1 problem
    '''
    # Define and solve the CVXPY problem.
    y_dic = [ff, y1, y2, y3]
    for j in ["inf", 1]:
        yhat_norm_inf = []
        print("\n\nThis part is for norm: ", j)
        for i in range(4):
            x = cp.Variable(N)
            cost = cp.norm(A @ x - y_dic[i], j)
            prob = cp.Problem(cp.Minimize(cost))
            prob.solve()
            yhat_norm_inf.append(A @ x.value)
            # Print result.
            print("The optimal value is", round(prob.value, 2))
            print("The optimal x is")
            print(list(round(i, 2) for i in x.value))
            print("The norm of the residual is ", round(cp.norm(A @ x - ff, p=j).value, 2))
            print('===========================================================================')

        fid = plt.figure(figsize=(10, 5))
        Axes = plt.subplot(1, 1, 1)
        Axes.axes.tick_params(which='both', direction='in', top=True, right=True)
        plt.minorticks_on()
        Axes.set_facecolor((0, 0, 0, 0.02))
        plt.plot(tt, ff, color='k', label='Origin f(t)')
        plt.plot(tt, yhat_norm_inf[0], 'k--', color='r', label='Polynomial fit')
        plt.plot(tt, yhat_norm_inf[1], 'k--', color='g', label='Random gaussian noise')
        plt.plot(tt, yhat_norm_inf[2], 'k--', color='b', label='Sparse gaussian noise')
        plt.plot(tt, yhat_norm_inf[3], 'k--', color='m', label='Uniform noise')
        plt.grid(True, which='major', linewidth=0.5)
        plt.grid(True, which='minor', linewidth=0.1)
        plt.xlabel('t')
        plt.title('Polynomial interpolation of f(t), Degree = {}, norm = {}'.format(N - 1, j))
        plt.legend(loc='upper right', fontsize='x-small')
        plt.show()
        # plt.savefig('/Users/gexueren/Desktop/6270/assignment/hw01/{}.jpg'.format(j))


