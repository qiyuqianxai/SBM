import cv2
import os
from keras.datasets import mnist

import numpy as np

train_pth = 'MNIST/train'
test_pth = 'MNIST/test'

# 自动下载mnist数据集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

for i in range(0, 59999):  # 迭代 0 到 59999 之间的数字
    os.makedirs(os.path.join(train_pth, str(Y_train[i])), exist_ok=True)
    fileName = os.path.join(train_pth,str(Y_train[i]),str(i)+".png")
    print(fileName)
    cv2.imwrite(fileName, X_train[i])

for i in range(0, 9999):  # 迭代 0 到 9999 之间的数字
    os.makedirs(os.path.join(test_pth,str(Y_test[i])),exist_ok=True)
    fileName = os.path.join(test_pth,str(Y_test[i]),str(i)+".png")
    cv2.imwrite(fileName, X_test[i])

