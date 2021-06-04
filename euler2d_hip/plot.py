import sys
import math
import numpy as np
import matplotlib.pyplot as plt

x, y, rho, vx, vy, pre = np.loadtxt(sys.argv[1]).T
ni, nj = [math.isqrt(len(x))] * 2
x, y, rho, vx, vy, pre = [a.reshape([ni, nj]) for a in [x, y, rho, vx, vy, pre]]

plt.imshow(rho)
plt.show()
