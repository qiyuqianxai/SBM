import os
import random
import torch
import numpy as np
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(2048)

def getallpics(path, imgs):
    filelist = os.listdir(path)
    for file in filelist:
        # print(file,filecount)
        if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith(
                '.jpeg') or file.lower().endswith('.bmp'):
            imgs.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            getallpics(os.path.join(path, file), imgs)
        else:
            pass

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# generate imgs by random imgs or classes
def generate_node_data(data_pth="", node_num = 15, random_imgs=True, avg_alloc=False):
    all_node_data = []
    # random allocate img to all nodes
    if random_imgs:
        all_imgs = []
        getallpics(data_pth, all_imgs)
        # average allocate img to all nodes
        avg_size = len(all_imgs)//node_num
        for i in range(node_num-1):
            if avg_alloc:
                data_size = avg_size
            else:
                data_size = random.randint(int(len(all_imgs)*0.08),int(len(all_imgs)*0.15))
            node_data = random.sample(all_imgs, data_size)
            all_imgs = list(set(all_imgs) - set(node_data))
            all_node_data.append(node_data)
        all_node_data.append(all_imgs)
    # random allocate classes to nodes
    else:
        all_classes = [os.path.join(data_pth,cls) for cls in os.listdir(data_pth)]
        avg_size = len(all_classes) // node_num
        for i in range(node_num-1):
            if avg_alloc:
                data_size = avg_size
            else:
                data_size = random.randint(int(len(all_classes) * 0.1), int(len(all_classes) * 0.8))
            node_data_clses = random.sample(all_classes, data_size)
            all_classes = list(set(all_classes) - set(node_data_clses))
            node_data_imgs = []
            for cls_pth in node_data_clses:
                getallpics(cls_pth,node_data_imgs)
            all_node_data.append(node_data_imgs)
        node_data_clses = all_classes
        node_data_imgs = []
        for cls_pth in node_data_clses:
            getallpics(cls_pth, node_data_imgs)
        all_node_data.append(node_data_imgs)
    for node_data in all_node_data:
        print(len(node_data))
    return all_node_data

def generate_test_data(data_pth):
    all_imgs = []
    getallpics(data_pth,all_imgs)
    return all_imgs

if __name__ == '__main__':
    generate_node_data(random_imgs = False)