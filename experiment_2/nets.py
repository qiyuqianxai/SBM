#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch as t
import math
from torchsummary import summary
import numpy as np


def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.3)
        nn.init.constant_(m.bias,0.1)

class DQN_Net(nn.Module):
    def __init__(self,s_dim, a_dim):
        super(DQN_Net, self).__init__()
        self.layer1 = nn.Linear(s_dim,64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64,a_dim)
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        return x

class CurConv2d(nn.Module):
    def __init__(self, channels, filters, ksize, isize, rows, cols):
        super(CurConv2d, self).__init__()
        k_w_rows = channels * ksize * ksize

        self.weight_c = nn.Parameter(t.Tensor(k_w_rows, cols))
        self.weight_r = nn.Parameter(t.Tensor(rows, filters))
        self.bias = nn.Parameter(t.Tensor(filters, ))

        self.unfold = nn.Unfold(ksize, dilation=1, padding=1, stride=1)
        self.fold = nn.Fold(isize, 1)

        self.rows = nn.Parameter(t.Tensor(rows, ), requires_grad=False)
        self.cols = nn.Parameter(t.Tensor(cols, ), requires_grad=False)

        a = t.empty([k_w_rows, filters])
        nn.init.xavier_uniform_(a)


        u, s, v = t.svd(a)
        s[rows:] = 0
        a = t.mm(t.mm(u, t.diag(s)), v.t())

        row_norm = t.norm(u, dim=1)
        _, row_topkindex = t.topk(row_norm, rows)
        rows = row_topkindex.numpy()

        col_norm = t.norm(v, dim=1)
        _, col_topkindex = t.topk(col_norm, cols)
        cols = col_topkindex.numpy()

        self.rows = nn.Parameter(t.from_numpy(rows).long(), requires_grad=False)
        self.cols = nn.Parameter(t.from_numpy(cols).long(), requires_grad=False)

        self.weight_c.data = a[:, self.cols.long()]
        self.weight_r.data = a[self.rows.long(), :]

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(t.Tensor(
            self.weight_c.size(0), self.weight_r.size(1)))
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        u = t.pinverse((self.weight_c[self.rows.long(), :] + self.weight_r[:, self.cols.long()]) / 2)
        unfolded_x = self.unfold(x)

        flat_x = unfolded_x.transpose(1, 2).contiguous().view(-1, self.weight_c.size(0))

        result = (flat_x.matmul(self.weight_c).matmul(u.matmul(
            self.weight_r)).view(x.size(0), -1, self.weight_r.size(1)) + self.bias).transpose(1, 2)

        return self.fold(result)

class CurLinear(nn.Module):
    def __init__(self, in_features, nodes, rows, cols):
        super(CurLinear, self).__init__()

        self.weight_c = nn.Parameter(t.Tensor(in_features, cols))
        self.weight_r = nn.Parameter(t.Tensor(rows, nodes))
        self.bias = nn.Parameter(t.Tensor(nodes, ))

        # initializations for weight c, r and bias.
        a = t.empty([self.weight_c.size(0), self.weight_r.size(1)])
        nn.init.xavier_uniform_(a)

        u, s, v = t.svd(a)
        s[rows:] = 0
        a = t.mm(t.mm(u, t.diag(s)), v.t())


        row_norm = t.norm(u, dim=1)
        _, row_topkindex = t.topk(row_norm, rows)
        rows = row_topkindex.numpy()
        # rows = np.sort(rows)

        col_norm = t.norm(v, dim=1)
        _, col_topkindex = t.topk(col_norm, cols)
        cols = col_topkindex.numpy()
        # cols = np.sort(cols)


        self.rows = nn.Parameter(t.from_numpy(rows).long(), requires_grad=False)
        self.cols = nn.Parameter(t.from_numpy(cols).long(), requires_grad=False)
        self.weight_c.data = a[:, self.cols]
        self.weight_r.data = a[self.rows, :]

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(t.Tensor(
            self.weight_c.size(0), self.weight_r.size(1)))
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        u = t.pinverse((self.weight_c[self.rows, :] + self.weight_r[:, self.cols]) / 2)
        return x.matmul(self.weight_c).matmul(u).matmul(self.weight_r) + \
               self.bias

class CURVGG8(nn.Module):
    def __init__(self, kerset, num_classes=10):
        super(CURVGG8, self).__init__()
        self.conv1 = CurConv2d(3, 64, 3, 32, kerset[0], kerset[1])
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.conv2 = CurConv2d(64, 128, 3, 16, kerset[2], kerset[3])
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=True)
        self.conv3 = CurConv2d(128, 256, 3, 8, kerset[4], kerset[5])
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=True)
        self.conv4 = CurConv2d(256, 256, 3, 8, kerset[6], kerset[7])
        self.bn4 = nn.BatchNorm2d(256, track_running_stats=True)
        self.conv5 = CurConv2d(256, 512, 3, 4, kerset[8], kerset[9])
        self.bn5 = nn.BatchNorm2d(512, track_running_stats=True)
        self.conv6 = CurConv2d(512, 512, 3, 4, kerset[10], kerset[11])
        self.bn6 = nn.BatchNorm2d(512, track_running_stats=True)
        self.conv7 = CurConv2d(512, 512, 3, 2, kerset[12], kerset[13])
        self.bn7 = nn.BatchNorm2d(512, track_running_stats=True)
        self.conv8 = CurConv2d(512, 512, 3, 2, kerset[14], kerset[15])
        self.bn8 = nn.BatchNorm2d(512, track_running_stats=True)
        self.fc1 = CurLinear(512, 4096, kerset[16], kerset[17])
        self.fc2 = CurLinear(4096, 4096, kerset[18], kerset[19])
        self.fc3 = nn.Linear(4096, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # x = self.pool(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        # x = self.pool(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, kerset, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = CurConv2d(3, 64, 3, 32, kerset[0], kerset[1])
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.conv2 = CurConv2d(64, 128, 3, 16, kerset[2], kerset[3])
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=True)
        self.conv3 = CurConv2d(128, 256, 3, 8, kerset[4], kerset[5])
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4096, num_classes)
        # self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNCifar_src(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar_src, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3,1,1)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4096, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




def calculate_ratio(x):
    all_num = 28092096
    w = 27 * x[1] + x[0] * 64
    w += 576 * x[3] + x[2] * 128
    w += 1152 * x[5] + x[4] * 256
    w += 2304 * x[7] + x[6] * 256
    w += 2304 * x[9] + x[8] * 512
    w += 4608 * x[11] + x[10] * 512
    w += 4608 * x[13] + x[12] * 512
    w += 4608 * x[15] + x[14] * 512
    w += 512 * x[17] + x[16] * 4096
    w += 4096 * x[19] + x[18] * 4096

    return 1 - w / all_num

if __name__ == '__main__':
    kersets = [
        [18, 30, 60, 45, 65, 50, 80, 55, 80, 55, 80, 35, 80, 35, 80, 35, 35, 50, 50, 50],
        [65,35,35,35]
    ]

    net = CNNCifar([18, 30, 60, 45, 65, 50])
    summary(net,(3,32,32))

    o_net = CNNCifar_src()
    summary(o_net,(3,32,32))


    # compress_ratio = calculate_ratio(kersets[1])
    print("compress_ratio:",1 - 42314/412682)