from torch.utils.data import Dataset,DataLoader
import torch
from utils import generate_node_data
import random
from torchvision.transforms import transforms
from PIL import Image
import os

class node_datasets(Dataset):
    def __init__(self,node_data,train=True):
        self.data = node_data
        self.is_train = train
        # self.train = random.sample(self.data,int(len(self.data)*0.8))
        # self.val = list(set(self.data)-set(self.train))
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item]).convert('RGB')
        img = self.img_transform(img)
        class_chars = os.listdir(os.path.sep.join(self.data[item].split(os.path.sep)[:-2]))
        label = class_chars.index(self.data[item].split(os.path.sep)[-2])
        # print(data[item],label)
        return img, label

    def transform(self):
        compose_train = [
            # transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            # transforms.CenterCrop(150),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        compose_dev = [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        if self.is_train:
            return transforms.Compose(compose_train)
        return transforms.Compose(compose_dev)

if __name__ == '__main__':
    database_root = "D:/cifar10"
    all_node_data = generate_node_data(random_imgs = False)
    dataset = node_datasets(all_node_data[0])
    dataloader = DataLoader(dataset,batch_size=8,shuffle=True)
    for batch in dataloader:
        x,y = batch
        print(x.shape[0],y.shape[0])