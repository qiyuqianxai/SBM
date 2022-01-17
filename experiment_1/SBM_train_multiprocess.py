import torch
from multiprocessing import Process,queues
import numpy as np
from dataset import node_datasets
from torch.utils.data import Dataset,DataLoader
from utils import same_seeds
from multiprocessing import get_context
from nets import CurRetrainingModel,CurModel,calculate_ratio
from torch import nn, optim
import random
import os
import redis
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
same_seeds(2048)

def getRedis():
    pool = redis.ConnectionPool(host="10.128.1.232", port="6381",password=123456,decode_responses=True, health_check_interval=30,db=1)
    r = redis.Redis(connection_pool=pool)
    r.ping()
    return r
r = getRedis()
# node train
def node_train(node_id, data, args,redis_db):
    # prepare node data
    node_train_datasets = node_datasets(data,train=True)
    node_train_dataloader = DataLoader(node_train_datasets,batch_size=args.batch_size, shuffle=True, num_workers=args.node_num_workers, pin_memory=True)

    node_val_datasets = node_datasets(data,train=False)
    node_val_dataloader = DataLoader(node_val_datasets,batch_size=args.batch_size,shuffle=False,num_workers=args.node_num_workers,pin_memory=True)

    print(f"node {node_id}: train data size:{node_train_datasets.__len__()}, val data size:{node_val_datasets.__len__()}")

    # train config
    # init model
    device = args.device
    last_model = CurModel(args.net_nodes,args.kerset)
    last_model.to(device)
    compression_ratio = calculate_ratio(args.net_nodes,args.kerset)
    print(f"node{node_id}:model param compression ratio: {compression_ratio:.2f}")
    node_lr = args.node_lr
    global_lr = args.global_lr
    criterion = nn.BCELoss()
    per_epoch_acc_record = []
    per_epoch_loss_record = []
    for epoch in range(args.iterations): # all nodes cycle epoches
        print(f"node{node_id} start epoch{epoch} training")
        # laod last-model data
        node_model = CurModel(args.net_nodes,args.kerset,last_model)
        node_model.to(device)
        optimizer = optim.SGD(node_model.parameters(), lr=node_lr, momentum=0.9, weight_decay=5e-4)
        last_model_optimizer = optim.SGD(last_model.parameters(), lr=global_lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                              gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

        # train:node learning
        node_model.train()
        for node_epoch in range(args.node_epochs):
            epoch_loss = []
            epoch_acc = []
            for x,y in node_train_dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                logits = node_model(x)
                loss = criterion(logits,y)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                acc = (torch.argmax(logits,dim=-1) == y).float().mean()
                epoch_acc.append(acc)
            scheduler.step()
            print(f"epoch {node_epoch}:node_model train_loss:{sum(epoch_loss)/len(epoch_loss)},train_acc:{sum(epoch_acc)/len(epoch_acc)}")

        epoch_loss = []
        epoch_acc = []
        node_model.eval()
        # val:get node model weight score
        tmp_acc_record = []
        tmp_loss_record = []
        with torch.no_grad():
            for x, y in node_val_dataloader:
                x = x.to(device)
                y = y.to(device)
                logits = node_model(x)
                loss = criterion(logits, y)
                epoch_loss.append(loss.item())
                acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                epoch_acc.append(acc)
        print(
            f"epoch {epoch}:node_model val_loss:{sum(epoch_loss) / len(epoch_loss)},val_acc:{sum(epoch_acc) / len(epoch_acc)}")
        tmp_acc_record.append(sum(epoch_acc) / len(epoch_acc))
        tmp_loss_record.append(sum(epoch_loss) / len(epoch_loss))
        # according to the display in val_data, get the gradient weigh of the node
        weight_score = (sum(epoch_acc) / len(epoch_acc))*node_val_datasets.__len__()
        print(f"node{node_id} weight score:{weight_score}")
        # record the beginning model param data to check the security
        last_model_param_data = []
        for param in last_model.parameters():
            last_model_param_data.append(param.data)
        # record the gradient
        node_model_grad = []
        # grad or weight data
        for param in node_model.parameters():
            node_model_grad.append(param.grad)

        # after training finish, broadcast param to other node
        info = {
            "node_id": node_id,
            "weight_score": weight_score,
            "last_model_param_data": last_model_param_data,
            "node_model_grad": node_model_grad
        }
        print(f"node{node_id} broadcast info...")
        for key in info:
            redis_db.hset(f"node{node_id}",key,info[key])
        print(f"node{node_id} broadcast info finish!")

        # wait until node msg queue receive all other node info
        while True:
            print(f"node{node_id} wait for other node info...")
            print(len(redis_db.keys()))
            if len(redis_db.keys()) == args.node_num:
                all_node_infos = []
                for h_name in redis_db.keys():
                    info = {}
                    for key in redis_db.hkeys(h_name):
                        info[key] = redis_db.hget(h_name,key)
                    all_node_infos.append(info)

                if args.need_security_check:
                    print(f"node{node_id} start security check...")
                    normal_node_infos = security_check(all_node_infos,args.security_threshold)
                else:
                    normal_node_infos = all_node_infos

                # merge all node's model grad
                print(f"node{node_id} start merge model")
                merge_grads = []
                scores = 0
                for node_info in normal_node_infos:
                    node_grad = node_info["node_model_grad"]
                    score = node_info["weight_score"]
                    scores += score
                    for i,grad_data in enumerate(node_grad):
                        if len(merge_grads) < len(node_grad):
                            merge_grads.append(grad_data*score)
                        else:
                            merge_grads[i] += grad_data*score
                print(f"node{node_id} merge model finish!")
                # update model
                last_model_optimizer.zero_grad()
                merge_grads = [grad_data/scores for grad_data in merge_grads]
                for last_param, merge_grad in zip(last_model.parameters(), merge_grads):
                    last_param.grad = merge_grad
                last_model_optimizer.step()
                print("node",node_id,"model update success!")
                break

        # save model weights
        os.makedirs(args.save_dir,exist_ok=True)
        if epoch%args.save_interval == 0 or epoch == args.iterations - 1:
            torch.save(last_model.state_dict(), os.path.join(args.save_dir,f"node_{node_id}-epoch_{epoch}.pth"))

        # test global model display in val_dataset
        epoch_loss = []
        epoch_acc = []
        last_model.eval()
        with torch.no_grad():
            for x, y in node_val_dataloader:
                x = x.to(device)
                y = y.to(device)
                logits = last_model(x)
                loss = criterion(logits, y)
                epoch_loss.append(loss.item())
                acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                epoch_acc.append(acc)
        print(
            f"epoch {epoch}:global_model val_loss:{sum(epoch_loss) / len(epoch_loss)},val_acc:{sum(epoch_acc) / len(epoch_acc)}")

        # record log
        tmp_acc_record.append(sum(epoch_acc) / len(epoch_acc))
        tmp_loss_record.append(sum(epoch_loss) / len(epoch_loss))
        per_epoch_acc_record.append(tmp_acc_record)
        per_epoch_loss_record.append(tmp_loss_record)
        os.makedirs(args.log_dir,exist_ok=True)
        save_pth = os.path.join(args.log_dir,f"node{str(node_id)}_loss_record.png")
        draw_plot(save_pth,per_epoch_loss_record,2,"loss record")
        save_pth = os.path.join(args.log_dir,f"node{str(node_id)}_acc_record.png")
        draw_plot(save_pth, per_epoch_acc_record, 1, "acc record")

def draw_plot(save_pth, records, y_scale, ylabel):
    label_list = range(len(records))  # 横坐标刻度显示值
    num_list1 = [record[0] for record in records]  # 纵坐标值1
    num_list2 = [record[1] for record in records]  # 纵坐标值2
    x = range(len(num_list1))

    rects1 = plt.bar(left=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="node_model")
    rects2 = plt.bar(left=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="global_model")
    plt.ylim(0, y_scale)  # y轴取值范围
    plt.ylabel(ylabel)
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("epoch")
    plt.title(ylabel+" vary record")
    plt.legend()  # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    # plt.show()
    plt.savefig(save_pth)

# check the consistence of all node's param
def security_check(all_node_infos,security_threshold):
    last_mode_params = []
    for info in all_node_infos:
        last_mode_params.append(info["last_model_param_data"])
    unique_params = set(last_mode_params)
    if len(unique_params) == 1:
        return all_node_infos
    else:
        noraml_infos = []
        for i,param in enumerate(last_mode_params):
            param_num = last_mode_params.count(param)
            # number more than 80% of total number, deem normal node data
            if param_num > len(last_mode_params)*security_threshold:
                noraml_infos.append(all_node_infos[i])
        return noraml_infos


def SBM(all_node_data,args):
    # every node get a queue to communicate
    redis_db = getRedis()
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(args.node_num)
    start_time = time.time()
    for i in range(args.node_num):
        # boot all node to train
        pool.apply_async(node_train,args=(i,all_node_data[i],args,redis_db,))
    # for P in all_node_processes:
    #     P.join()
    pool.close()
    pool.join()
    end_time = time.time()
    print(f"SBM train finish! cost time:{end_time-start_time}s")
