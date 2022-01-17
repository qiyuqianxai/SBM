import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.image as plimg

CHANNEL = 3
WIDTH = 32
HEIGHT = 32

data = []
labels = []
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# for i in range(5):
#     with open("./cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
#         # 数据集在当脚本前文件夹下
#         data_dict = pickle.load(file, encoding='bytes')
#         data += list(data_dict[b'data'])
#         labels += list(data_dict[b'labels'])
#
# img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])
#
# # 代码创建文件夹，也可以自行创建
# data_path = "./dataset/cifar-10/"
#
# for i in range(img.shape[0]):
#     r = img[i][0]
#     g = img[i][1]
#     b = img[i][2]
#
#     # plimg.imsave("./pic4/" + str(i) + "r" + ".png", r)
#     # plimg.imsave("./pic4/" + str(i) + "g" + ".png", g)
#     # plimg.imsave("./pic4/" + str(i) + "b" + ".png", b)
#
#     ir = Image.fromarray(r)
#     ig = Image.fromarray(g)
#     ib = Image.fromarray(b)
#     rgb = Image.merge("RGB", (ir, ig, ib))
#
#     save_pth = os.path.join(data_path,classification[labels[i]])
#     os.makedirs(save_pth,exist_ok=True)
#     print(save_pth)
#     rgb.save(os.path.join(save_pth,str(i)+ ".png"), "PNG")

with open("./cifar-10-batches-py/test_batch", mode='rb') as file:
    # 数据集在当脚本前文件夹下
    data_dict = pickle.load(file, encoding='bytes')
    data += list(data_dict[b'data'])
    labels += list(data_dict[b'labels'])

img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

# 代码创建文件夹，也可以自行创建
data_path = "./dataset/cifar-10_test/"

for i in range(img.shape[0]):
    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    # plimg.imsave("./pic4/" + str(i) + "r" + ".png", r)
    # plimg.imsave("./pic4/" + str(i) + "g" + ".png", g)
    # plimg.imsave("./pic4/" + str(i) + "b" + ".png", b)

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    save_pth = os.path.join(data_path,classification[labels[i]])
    os.makedirs(save_pth,exist_ok=True)
    print(save_pth)
    rgb.save(os.path.join(save_pth,str(i)+ ".png"), "PNG")