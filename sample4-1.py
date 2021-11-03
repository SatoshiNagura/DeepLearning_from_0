import numpy as np
import matplotlib.pyplot as plt

def Function1(X):
    return np.reshape(X, (len(X), 1)) ** 2 + X ** 2

#中央差分で勾配を計算する
def Gradient(Func, delta):
    #列ベクトルの引き算で,x0方向の変化率を計算
    Column_m = Func[2:, :-2]#3 ~ n行、1 ~ n-2列
    Column_p = Func[2:, 2:]#3 ~ n行、3 ~ n列
    dx0 = Column_p - Column_m
    dx0 /= -delta

    #行ベクトルの引き算で、x1方向の変化率を計算
    Row_m = Func[:-2, 2:]#1 ~ n-2行、3 ~ n列
    Row_p = Func[2:, 2:]#3 ~ n行、3 ~ n列
    dx1 = Row_p - Row_m
    dx1 /= -delta

    #それぞれの方向の変化率を場所ごとにベクトルとして返す
    #return np.dstack((dx0, dx1))
    return np.array((dx0, dx1))
x = np.array((np.arange(-4.0, 4.5, 0.5)))
dF = Gradient(Function1(x), 0.5)

plt.figure()
plt.quiver(dF[0], dF[1])
plt.draw()
plt.show()
