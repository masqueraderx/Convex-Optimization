import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp

# Define function
f_opt = lambda t: np.cos(2*np.pi*t)


# Number of samples
M = 100

# Degree + 1
N = 9

# Choose sample location
tt = np.linspace(-1, 1, M)

# Compute "clean" function
ff = f_opt(tt)

# Form "observation matrix"
A = np.zeros((M,N))
for i in np.arange(N):
    A[:,i] = tt ** i
    

# Compute and plot least squares polynomial fit for "noise free" samples
xhat = np.linalg.pinv(A) @ ff
fhat = A @ xhat

plt.figure()
plt.plot(tt, ff, 'b')
plt.plot(tt, fhat, 'r--' )
plt.xlabel('t')
plt.title('Polynomial interpolation of f(t), Degree = {}'.format(N-1))
plt.legend(('Original f(t)','Polynomial fit'))


## Construct noisy observations
np.random.seed(2020) # Set random seed so results are repeatable

# Random gaussian noise
noise1 = np.random.randn(M)
noise1 = 5*noise1/np.linalg.norm(noise1)
y1 = ff + noise1

# Sparse gaussian noise
K = 15
Gamma = np.random.choice(M,K,0) 
noise2 = np.zeros(M)
noise2[Gamma] = np.random.randn(K)
noise2 = 5*noise2/np.linalg.norm(noise2)
y2 = ff + noise2

# Uniform noise
noise3 = np.random.rand(M)-0.5
noise3 = 5*noise3/np.linalg.norm(noise3)
y3 = ff + noise3


## Compute least squares fit from noisy samples
xhat = np.linalg.pinv(A) @ y1
fhat1_2 = A @ xhat
xhat = np.linalg.pinv(A) @ y2
fhat2_2 = A @ xhat
xhat = np.linalg.pinv(A) @ y3
fhat3_2 = A @ xhat

## Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
fig. suptitle('Least squares ($\ell_2$) estimates')
ax1.plot(tt, ff, 'b')
ax1.plot(tt, fhat1_2, 'r--')
ax1.scatter(tt, y1,s=1)
ax1.set(ylabel='Gaussian Noise')
ax2.plot(tt, ff, 'b')
ax2.plot(tt, fhat2_2, 'r--')
ax2.scatter(tt,y2,s=1)
ax2.set(ylabel='Sparse Noise')
ax3.plot(tt, ff, 'b')
ax3.plot(tt, fhat3_2, 'r--')
ax3.scatter(tt,y3,s=1)
ax3.set(ylabel='Uniform Noise')
plt.savefig('P8_L2.pdf')

## Compute Linfty fit from noisy samples
x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y1,'inf'))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat1_inf = A @ xhat

x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y2,'inf'))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat2_inf = A @ xhat

x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y3,'inf'))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat3_inf = A @ xhat

## Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
fig. suptitle('$\ell_\infty$ estimates')
ax1.plot(tt, ff, 'b')
ax1.plot(tt, fhat1_inf, 'r--')
ax1.scatter(tt, y1,s=1)
ax1.set(ylabel='Gaussian Noise')
ax2.plot(tt, ff, 'b')
ax2.plot(tt, fhat2_inf, 'r--')
ax2.scatter(tt,y2,s=1)
ax2.set(ylabel='Sparse Noise')
ax3.plot(tt, ff, 'b')
ax3.plot(tt, fhat3_inf, 'r--')
ax3.scatter(tt,y3,s=1)
ax3.set(ylabel='Uniform Noise')
plt.savefig('P8_Linf.pdf')

## Compute L1 fit from noisy samples
x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y1,1))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat1_1 = A @ xhat

x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y2,1))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat2_1 = A @ xhat

x = cp.Variable(N)
objective = cp.Minimize(cp.norm(A @ x - y3,1))
prob = cp.Problem(objective)
prob.solve()
xhat = x.value
fhat3_1 = A @ xhat

## Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
fig. suptitle('$\ell_1$ estimates')
ax1.plot(tt, ff, 'b')
ax1.plot(tt, fhat1_1, 'r--')
ax1.scatter(tt, y1,s=1)
ax1.set(ylabel='Gaussian Noise')
ax2.plot(tt, ff, 'b')
ax2.plot(tt, fhat2_1, 'r--')
ax2.scatter(tt,y2,s=1)
ax2.set(ylabel='Sparse Noise')
ax3.plot(tt, ff, 'b')
ax3.plot(tt, fhat3_1, 'r--')
ax3.scatter(tt,y3,s=1)
ax3.set(ylabel='Uniform Noise')
plt.savefig('P8_L1.pdf')
