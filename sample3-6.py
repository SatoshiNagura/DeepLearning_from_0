from tensorflow.keras.datasets import mnist #tensorflowでMNISTのデータを抽出するライブラリ
import cv2
import numpy as np
import pickle

#シグモイド関数の定義
def SigmoidFunction(X):
    return 1 / (1 + np.exp(-X))
#ソフトマックス関数の定義
def SoftmaxFuction(X):
    Max = np.max(X)
    return np.exp(X - Max) / np.sum(np.exp(X - Max))
#ネットワークの定義
def PredictFunction(Network, X):
    #ニューラルの重みとバイアスを受け取る
    W1, W2, W3 = Network['W1'], Network['W2'], Network['W3']
    B1, B2, B3 = Network['b1'], Network['b2'], Network['b3']
    #行列計算して、中間層と出力層の値を出す
    A1 = X @ W1 + B1
    Z1 = SigmoidFunction(A1)
    A2 = Z1 @ W2 + B2
    Z2 = SigmoidFunction(A2)
    A3 = Z2 @ W3 + B3
    return SoftmaxFuction(A3)

#予め保存してあるニューラルの重みとバイアスを読み出す
with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

#MNISTの教師データを読み出して、ベクトルに変換
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_img = np.reshape(test_images, (10000, 784))

#正答率の値を初期化
accuracy_cnt = 0

#1枚ごとにニューラルネットワークにぶちこむ
for i in range(10000):
    #出力層の値を取得
    y = PredictFunction(network, test_img[i])
    p = np.argmax(y)
    if p == test_labels[i]:
        accuracy_cnt += 1
print("正答率: " + str(float(accuracy_cnt) / 10000))
