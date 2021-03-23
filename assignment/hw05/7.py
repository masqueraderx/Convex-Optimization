import numpy as np
from matplotlib import pyplot as plt

def func(x):
    return x**2 + 1

def contraint(x):
    return (x-2) * (x-4)

def lagrange(x, lam):
    return func(x) + lam * contraint(x)

def dual(x):
    numerator = -x**3 + 8 * x**2 + 10 * x + 1
    denumerator = (x + 1)**2
    return numerator / denumerator

def plotObj(x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x, np.zeros(len(x)), 'bo', label='feasible set')
    plt.plot(x, y, 'r', label='objective function')
    plt.title('objective function')
    plt.xlim([1.0, 5.0])
    plt.ylim([-0.01, 20])
    plt.legend()
    plt.savefig('./hw05/7c.jpg')

def plotLagrange(x, y, lam):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'g', label='Lagrangian')
    plt.title('Lagrangian with lambda {}'.format(lam))
    plt.legend()
    plt.savefig('./hw05/7c-lam{}.jpg'.format(lam))

def plotDual(x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'y', label='Dual function')
    plt.title('Dual function')
    plt.legend()
    plt.savefig('./hw05/7d.jpg')


if __name__ == '__main__':
    x = np.linspace(2,4,100)
    y = func(x)
    plotObj(x, y)
    for lam in [0, 5, 10]:
        L = lagrange(x, lam)
        plotLagrange(x, L, lam)
    d = dual(x)
    plotDual(x, d)

