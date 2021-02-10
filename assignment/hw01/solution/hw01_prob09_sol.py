import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

## PROBLEM 9
# Construct the problem
N = 1000 # Population size
S = 10 # Number of infected individuals
M = 100 # Number of tests
K = 10 # Number of splits of each sample

## Construct noisy observations
np.random.seed(2020) # Set random seed so results are repeatable

# Define x0
ind0 = np.sort(np.random.choice(N,S,0)) # index subset, sorted in increasing order 
x0 = np.zeros(N) 
x0[ind0] = np.random.rand(S)

# Define A
A = np.zeros((M,N))
for i in np.arange(N):
    ind = np.random.choice(M,K,0)
    A[ind,i] = 1

y = A @ x0


## Set up recovery problem

x = cp.Variable(N)
objective = cp.Minimize(cp.norm(x,1))
constraints = [A@x == y, 0 <= x]
prob = cp.Problem(objective, constraints)

result = prob.solve()
xhat = x.value

indhat = np.argwhere(xhat>1e-9).flatten() # Can verify that this matches indo0

if np.array_equal(indhat,ind0):
    print('Infected individuals identified')
else:
    print('Incorrect individuals identified')

#plt.plot(x0,'b')
#plt.plot(xhat,'r--')


## Vary K
Pe = np.zeros(10)
for K in np.arange(1,11):
    
    errors = np.zeros(100)
    
    for trial in np.arange(100):
        A = np.zeros((M,N))
        for i in np.arange(N):
            ind = np.random.choice(M,K,0)
            A[ind,i] = 1

        y = A @ x0
        
        x = cp.Variable(N)
        objective = cp.Minimize(cp.norm(x,1))
        constraints = [A@x == y, 0 <= x]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        xhat = x.value
        indhat = np.argwhere(xhat>1e-9).flatten() # Can verify that this matches indo0

        if (not np.array_equal(indhat,ind0)):
            errors[trial-1] = 1
            
     
    print('K iter')        
    Pe[K-1] = np.mean(errors)
    
plt.figure()
plt.stem(np.arange(1,11),Pe)
plt.title('Probability of error')
plt.xlabel('K')
plt.show()
# plt.savefig('P9b.pdf')

K = 10

