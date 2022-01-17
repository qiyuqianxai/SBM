import torch
import numpy as np
from dataset import node_datasets
from torch.utils.data import Dataset,DataLoader
from utils import *
from torch import nn, optim
import random
import os
import copy
import time
from settings import get_energy_tp_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib
import json
from settings import get_center_node
from nets import DQN_Net
from ea import NodeChoose

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
same_seeds(2048)

# node train
def node_train(node_id, node_model, data, test_data, args):
    # record the beginning model param data to check the security
    # last_model_param = copy.deepcopy(node_model.state_dict())
    # prepare node data
    node_train_datasets = node_datasets(data, train=True)
    node_train_dataloader = DataLoader(node_train_datasets,batch_size=args.batch_size, shuffle=True, num_workers=args.node_num_workers,pin_memory=True)


    print(f"node{node_id}: train data size:{node_train_datasets.__len__()}")

    # train config
    # init model
    device = args.device

    node_lr = args.node_lr
    criterion = nn.CrossEntropyLoss()
    
    # laod last-model data
    # optimizer = optim.SGD(node_model.parameters(), lr=node_lr, momentum=0.9, weight_decay=5e-4)
    
    optimizer = optim.AdamW(node_model.parameters(), lr=node_lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

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
            # print(loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            acc = (torch.argmax(logits,dim=-1) == y).float().mean()
            epoch_acc.append(acc)
        scheduler.step()
        print(f"node{node_id} epoch{node_epoch}:node_model train_loss:{sum(epoch_loss)/len(epoch_loss)},train_acc:{sum(epoch_acc)/len(epoch_acc)}")


    node_model.eval()
    # test node on test-dataset
    test_dataset = node_datasets(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.node_num_workers,
                                 pin_memory=True)
    with torch.no_grad():
        epoch_loss = []
        epoch_acc = []
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = node_model(x)
            loss = criterion(logits, y)
            epoch_loss.append(loss.item())
            acc = (torch.argmax(logits, dim=-1) == y).float().mean()
            epoch_acc.append(acc)
        print(
            f"node{node_id}_model on test-dataset: loss:{sum(epoch_loss) / len(epoch_loss)},acc:{sum(epoch_acc) / len(epoch_acc)}")

    # according to the display in val_data, get the gradient weigh of the node
    weight_score = 1/(sum(epoch_loss) / len(epoch_loss)) * node_train_datasets.__len__()
    # energy_cp = node_train_datasets.__len__()*args.cp_ratio
    print(f"node{node_id} weight score:{weight_score}")

    # record the model param
    node_model_data = copy.deepcopy(node_model.state_dict())

    # after training finish, broadcast param to other node
    info = {
        "node_id": node_id,
        "weight_score": weight_score,
        # "last_model_param_data": last_model_param,
        "node_model_data": node_model_data,
    }



    return info,sum(epoch_loss) / len(epoch_loss),sum(epoch_acc) / len(epoch_acc)

def draw_plot(save_pth, records, y_scale, ylabel):
    plt.figure()
    label_list = range(len(records))  # 横坐标刻度显示值
    num_list1 = [round(float(record),4) for record in records]  # 纵坐标值1
    x = [f"{i}" for i in range(len(records)-1)]
    x.append("center")

    rects1 = plt.bar(x, num_list1, width=0.4, alpha=0.8, color='green')
    plt.ylim(0, y_scale)  # y轴取值范围
    plt.ylabel(ylabel)
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index for index in range(len(x))], x)
    plt.xlabel("node_id")
    plt.title(ylabel+" vary record")
    # plt.legend()  # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2,  height*0.8, f"{height:.2f}", ha="center", va="bottom")
    # plt.show()
    plt.savefig(save_pth)

@torch.no_grad()
def SBM_test(args, model, test_data):
    device = args.device
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_dataset = node_datasets(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.node_num_workers,
                                 pin_memory=True)
    epoch_loss = []
    epoch_acc = []
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        epoch_loss.append(loss.item())
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        acc = acc.cpu().numpy()
        epoch_acc.append(acc)
    print(
        f"global-model on test-dataset:loss:{sum(epoch_loss) / len(epoch_loss)},acc:{sum(epoch_acc) / len(epoch_acc)}")

    return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

def SBM(model, all_node_data, test_data, args):
    # every node get a queue to communicate
    start_time = time.time()
    a_ratio = 5
    tp_matrix = get_energy_tp_matrix(args.node_num,args.tp_ratio)
    cp_matrix = np.array([len(node_data)*args.cp_ratio for node_data in all_node_data])
    print(cp_matrix)
    ef_matrix = np.zeros(args.node_num)
    rl_observation = np.zeros(args.node_num)
    cost_energy_record = []
    loss_record = []
    acc_record = []
    total_energy_cost = 0
    if args.select_method == "rl":
        rl_model = DQN_Net(args.node_num, args.node_num)
        rl_model.load_state_dict(
            torch.load(args.rl_weights, map_location=lambda storage, loc: storage))
        rl_model.to(args.device)
        rl_model.eval()


    for iteration in range(args.iterations):
        # 使用DDQN筛选节点
        if args.select_method == "rl":
            observation = rl_observation[np.newaxis, :]
            observation = torch.tensor(observation, dtype=torch.float).to(args.device)
            t = time.time()
            with torch.no_grad():
                actions_value = rl_model(observation)
                values, select_nodes = torch.topk(actions_value, args.select_node_num)
                select_nodes = select_nodes.cpu().numpy()[0]
            print("rl time:",time.time()-t)
            if iteration > 0 and np.random.uniform() > acc_record[-1]*a_ratio/cost_energy_record[-1]:  # choosing action
                print("random prob:", acc_record[-1] * a_ratio / cost_energy_record[-1])
                select_nodes = random.sample(range(args.node_num), args.select_node_num)
        elif args.select_method == "ea":
            # 使用差分进化算法
            try:
                select_nodes = NodeChoose(cp_matrix,tp_matrix,ef_matrix,args.node_num,args.select_node_num,args.b_ratio)
            except Exception as e:
                print(e,ef_matrix)

            if iteration > 0 and np.random.uniform() > acc_record[-1]*a_ratio/cost_energy_record[-1]:  # choosing action
                print("random prob:", acc_record[-1] * a_ratio / cost_energy_record[-1])
                select_nodes = random.sample(range(args.node_num), args.select_node_num)

        elif args.select_method == "random":
            # 完全随机选取节点
            select_nodes = random.sample(range(args.node_num), args.select_node_num)
        elif args.select_method == None:
            # 不筛选节点,即使用FL
            select_nodes = list(range(args.node_num))

        print("#"*50+f" iteration{iteration} "+"#"*50)
        center_node, all_enery_tp = get_center_node(select_nodes, tp_matrix)
        print(f"selected nodes:{select_nodes},center node:{center_node}")
        per_iter_loss_record = []
        per_iter_acc_record = []
        per_iter_nodes_res = []

        # nodes train
        for i in range(args.node_num):
            if i not in select_nodes:
                per_iter_acc_record.append(0)
                per_iter_loss_record.append(0)
                continue
            # boot all node to train
            info,tmp_loss_record,tmp_acc_record = node_train(i,copy.deepcopy(model).to(args.device),all_node_data[i], test_data, args)
            per_iter_acc_record.append(tmp_acc_record)
            per_iter_loss_record.append(tmp_loss_record)
            per_iter_nodes_res.append(info)
            ef_matrix[i] = info["weight_score"]/(cp_matrix[i]+tp_matrix[center_node,i])
            rl_observation[i] = info["weight_score"]/(cp_matrix[i]+tp_matrix[center_node,i])

        rl_observation = standardization(rl_observation)
        # merge all node's model data
        print("start merge all node model param")

        if args.select_method == None:
            # 使用Fedavg
            scores = sum([len(all_node_data[node_info["node_id"]]) for node_info in per_iter_nodes_res])
            w_avg = copy.deepcopy(per_iter_nodes_res[0]["node_model_data"])
            for k in w_avg.keys():
                if 'rows' in k or 'cols' in k or 'num_batches_tracked' in k:
                    continue
                w_avg[k] = w_avg[k] * len(all_node_data[per_iter_nodes_res[0]["node_id"]])
                for i in range(1, len(per_iter_nodes_res)):
                    w_avg[k] += per_iter_nodes_res[i]["node_model_data"][k] * len(all_node_data[per_iter_nodes_res[i]["node_id"]])
                w_avg[k] = torch.div(w_avg[k], scores)
        else:
            scores = sum([node_info["weight_score"] for node_info in per_iter_nodes_res])
            w_avg = copy.deepcopy(per_iter_nodes_res[0]["node_model_data"])
            for k in w_avg.keys():
                if 'rows' in k or 'cols' in k or 'num_batches_tracked' in k:
                    continue
                w_avg[k] = w_avg[k] * per_iter_nodes_res[0]["weight_score"]
                for i in range(1, len(per_iter_nodes_res)):
                    w_avg[k] += per_iter_nodes_res[i]["node_model_data"][k]*per_iter_nodes_res[i]["weight_score"]
                w_avg[k] = torch.div(w_avg[k], scores)
        model.load_state_dict(w_avg)
        print("merge model finish!")
        
        # test model on all node val-datasets
        per_iter_test_loss_record, per_iter_test_acc_record = SBM_test(args,copy.deepcopy(model).to(args.device),test_data)

        # record log
        loss_record.append(per_iter_test_loss_record)
        acc_record.append(per_iter_test_acc_record)

        per_iter_loss_record.append(per_iter_test_loss_record)
        per_iter_acc_record.append(per_iter_test_acc_record)
        os.makedirs(args.log_dir,exist_ok=True)
        save_pth = os.path.join(args.log_dir,f"iter{iteration}_loss_record.png")
        draw_plot(save_pth,per_iter_loss_record, int(max(per_iter_loss_record))+1,"loss record")
        save_pth = os.path.join(args.log_dir,f"iter{iteration}_acc_record.png")
        draw_plot(save_pth, per_iter_acc_record, 1, "acc record")

        # compute energy_cp, energy_tp
        all_enery_cp = np.sum(cp_matrix[select_nodes])
        total_energy_cost += all_enery_cp + all_enery_tp
        print(f"iteration{iteration}:cost energy:{all_enery_cp + all_enery_tp}")
        cost_energy_record.append(all_enery_cp + all_enery_tp)
        # save model
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"SBM_iter{iteration}.pth"))

        plt.figure()
        plt.plot(np.array(cost_energy_record), c='g', label='energy cost')
        plt.legend(loc='best')
        plt.ylabel('energy_value')
        plt.xlabel('iteration')
        plt.grid()
        plt.savefig(f"total_E_{args.select_method}.png")
        # plt.show()

        plt.figure()
        plt.plot(np.array(loss_record), c='y', label='global loss')
        plt.legend(loc='best')
        plt.ylabel('loss_value')
        plt.xlabel('iteration')
        plt.grid()
        plt.savefig(f"total_Loss_{args.select_method}.png")

        plt.figure()
        plt.plot(np.array(acc_record), c='r', label='global accuracy')
        plt.legend(loc='best')
        plt.ylabel('acc_value')
        plt.xlabel('iteration')
        plt.grid()
        plt.savefig(f"total_Acc_{args.select_method}.png")
    res = {"E_cost":cost_energy_record,
           "loss":loss_record,
           "acc":acc_record}
    print(res)
    with open(f"result_{args.select_method}.json","w",encoding="utf-8")as f:
        f.write(json.dumps(res,indent=4,ensure_ascii=False))
    end_time = time.time()
    print(f"SBM train finish! cost time:{end_time-start_time}s")
    print(f"total cost energy:{total_energy_cost}")


if __name__ == '__main__':
    draw_plot("./test.png",[0.5,0.7,0.7,0.89,0.43567],1,"test")