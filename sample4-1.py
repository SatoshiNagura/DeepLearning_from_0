import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Function1(X):
    return np.reshape(X, (81, 1)) ** 2 + X ** 2

x = np.array((np.arange(-4.0, 4.1, 0.1)))
z = Function1(x)

print(z)
"""
Axes3D.plot_wireframe(x[0], x[1], z, color = 'blue')
plt.show()"""
