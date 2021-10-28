import numpy as np
import matplotlib.pyplot as plt

def Sigmoid_Function(X):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
#a = x[2:3] #表示は値でも渡されるのはアドレス
y = Sigmoid_Function(x)
plt.plot(x, y)
plt.show()
