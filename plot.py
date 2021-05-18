import numpy as np
import matplotlib.pyplot as plt

x, rho, vx, pre = np.loadtxt('euler.dat').T
plt.plot(x, rho, label='density')
plt.plot(x, pre, label='pressure')
plt.legend()
plt.show()
