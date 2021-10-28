import numpy as np
import matplotlib.pyplot as plt

#numpy配列でも使えるシグモイド関数を定義
def Sigmoid_Fuction(X):
    return 1 / (1 + np.exp(-X))

x = np.array(input().split(", "), dtype = "float").T
arr = np.array(((1/3, 1/3, 1/3), (1, 0, 0), (1/2, 1, -1/2)))
middle = arr @ x
y = Sigmoid_Fuction(middle)
print(y)
