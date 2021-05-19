import sys
import numpy as np
import matplotlib.pyplot as plt

x, rho, vx, pre = np.loadtxt(sys.argv[1]).T
plt.plot(x, rho, label='density')
plt.plot(x, pre, label='pressure')
plt.legend()
plt.show()
