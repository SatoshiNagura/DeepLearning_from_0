import tensorflow as tf
from tensorflow.keras.datasets import mnist #tensorflowでMNISTのデータを抽出するライブラリ
import cv2
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
