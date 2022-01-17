
import os
import time

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, RMSprop
import random
import matplotlib.pyplot as plt
from SBM_train import *
from utils import *
from settings import get_energy_tp_matrix,get_center_node
from nets import *
import argparse
torch.cuda.set_device(1)

def get_args():
    parser = argparse.ArgumentParser()
    # global arguments
    parser.add_argument('--node_num', type=int, default=20,
                        help='all node numbers')
    parser.add_argument('--select_node_num',type=int,default=5,
                        help='participate training node numbers')
    parser.add_argument('--train_data_pth', type=str, default='dataset/cifar-10',
                        help='train_data_pth')
    parser.add_argument('--test_data_pth', type=str, default='dataset/cifar-10_test',
                        help='test_data_pth')
    parser.add_argument('--num_classes', type=int, default=10,
                        help="the classes of the dataset's class")
    parser.add_argument('--random_imgs', default=True, action='store_true',
                        help='decide allocate data to nodes by random imgs or classes')
    parser.add_argument('--avg_alloc', default=False,
                        help='decide the allocation of data is average or not', action='store_true')
    parser.add_argument('--select_method', type=str, default='rl',
                        help='chose the method of select nodes,ea or rl or random or None')
    parser.add_argument('--rl_weights', type=str, default='checkpoints/rl_episode7.pth',
                        help='the path of weights of rl_model')

    parser.add_argument('--kerset', type=list, default=[18, 30, 60, 45, 65, 50],
                        help='the kerset of Rolser funtion')
    parser.add_argument('--cp_ratio', type=float, default=0.0005, help='energy of compute ratio')
    parser.add_argument('--tp_ratio', type=float, default=0.1, help='energy of transport ratio')
    parser.add_argument('--b_ratio', type=float, default=0.9, help='sensitive of ef')

    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='the path of save weights of model')
    parser.add_argument('--log_dir', type=str, default='logs')

    parser.add_argument('--pretrained_weight', type=str, default='',
                        help='pretrained model weight path')

    # node_train_config
    parser.add_argument('--iterations', type=int, default=60,
                        help='global async epochs')
    parser.add_argument('--node_epochs', type=int, default=5,
                        help='every node training epochs')
    parser.add_argument('--node_lr', type=float, default=1e-3,
                        help='every node_model learning rate')
    parser.add_argument('--node_num_workers', type=int, default=4,
                        help="node dataloader's numworkers")
    parser.add_argument('--batch_size', type=int, default=100,
                        help='dataloader batch size of nodes')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def train(args):
    train_data = []
    getallpics(args.train_data_pth, train_data)
    test_data = []
    getallpics(args.test_data_pth, test_data)


    train_datasets = node_datasets(train_data, train=True)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.node_num_workers, pin_memory=True)
    print(f"train data size:{train_datasets.__len__()}")

    # test node on test-dataset
    test_datasets = node_datasets(test_data)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.node_num_workers,
                                 pin_memory=True)
    print(f"test data size:{test_datasets.__len__()}")


    # train config
    # init model
    model = CNNCifar(args.kerset, 10).to(args.device)
    node_lr = args.node_lr
    criterion = nn.CrossEntropyLoss()

    # laod last-model data
    # optimizer = optim.SGD(node_model.parameters(), lr=node_lr, momentum=0.9, weight_decay=5e-4)

    optimizer = optim.AdamW(model.parameters(), lr=node_lr, weight_decay=5e-4)

    loss_record = []
    acc_record = []
    # train:node learning
    for epoch in range(args.node_epochs*args.iterations):
        model.train()
        epoch_loss = []
        epoch_acc = []
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            loss = criterion(logits, y)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            acc = (torch.argmax(logits, dim=-1) == y).float().mean()
            epoch_acc.append(acc)
        print(
            f"epoch{epoch}:model train_loss:{sum(epoch_loss) / len(epoch_loss)},train_acc:{sum(epoch_acc) / len(epoch_acc)}")
        if (epoch+1)%args.node_epochs==0:
            model.eval()

            with torch.no_grad():
                epoch_loss = []
                epoch_acc = []
                for x, y in test_dataloader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    epoch_loss.append(loss.item())
                    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                    epoch_acc.append(acc.cpu().numpy())
                print(
                    f"model on test-dataset: loss:{sum(epoch_loss) / len(epoch_loss)},acc:{sum(epoch_acc) / len(epoch_acc)}")
                loss_record.append(sum(epoch_loss) / len(epoch_loss))
                acc_record.append(sum(epoch_acc) / len(epoch_acc))

                plt.figure()
                plt.plot(np.array(loss_record), c='b', label='loss on test-data')
                plt.legend(loc='best')
                plt.ylabel('loss_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig("Loss_centralization.png")

                plt.figure()
                plt.plot(np.array(acc_record), c='y', label='acc on test-data')
                plt.legend(loc='best')
                plt.ylabel('acc_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig("Acc_centralization.png")
    print("loss record:", loss_record)
    print("acc record", acc_record)
    with open("central_res.json","w",encoding="utf-8")as f:
        f.write(json.dumps({
            "loss record:":loss_record,
            "acc record":acc_record},indent=4,ensure_ascii=False))


if __name__ == '__main__':
    args = get_args()
    train(args)