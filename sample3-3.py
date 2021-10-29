import numpy as np

#numpy配列でも使えるシグモイド関数を定義
def Sigmoid_Fuction(X):
    return 1 / (1 + np.exp(-X))

#入力層とバイアス、重み付けの初期化
x = np.array((1, 0.5, 1))#入力層とバイアス
w = np.array(((0.1, 0.3, 0.5),#第一入力値に対する重み
              (0.2, 0.4, 0.6),
              (0.1, 0.2, 0.3)))#バイアスに対する重み
a = x @ w
z = Sigmoid_Fuction(a)
print(a)
print(z)
