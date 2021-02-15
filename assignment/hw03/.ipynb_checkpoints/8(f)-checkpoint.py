import numpy as np
from sklearn import datasets
import math

if __name__ == '__main__':
    np.random.seed(2020) # Set random seed so results are repeatable
    x, y = datasets.make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=6.0)
    x_tilde = x.tolist()
    x_tilde.append([1, 1])
    x_tilde = np.array(x_tilde)
    alpha = 0.001
    #set initial guess
    theta, k = np.array([0, 0]), 0

    # while:
    #     #calculate a direction to move dk
        for i in range(len(x_tilde)):
            dk += pow(math.e, theta * x_tilde)
    #     theta = theta - alpha * dk
    #     k = k + 1