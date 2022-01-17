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


class Double_DQN():
    def __init__(self, args):
        self.s_dim = args.node_num
        self.a_dim = args.node_num
        self.eval_net = DQN_Net(self.s_dim,self.a_dim).to(args.device)
        self.target_net = DQN_Net(self.s_dim,self.a_dim).to(args.device)
        self.save_pth = args.rl_checkpoints
        self.args = args
        self.epsilon_max = self.args.e_greedy
        self.epsilon = 0 if self.args.e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.args.memory_size, self.s_dim * 2 + self.args.select_node_num + 1))
        self.critisen = nn.MSELoss()
        self.optimizer = RMSprop(self.eval_net.parameters(), lr=self.args.lr, momentum=0.9)

    def learn(self):
        # 更新target模型参数
        if self.learn_step_counter+1 % self.args.replace_target_iter == 0:
            self.target_net.load_state_dict(torch.load(self.eval_net.state_dict()))
            # 增加epsilon
            self.epsilon = self.epsilon + self.args.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # 取一个batch的数据对eval_net进行训练
        if self.memory_counter > self.args.memory_size:
            sample_index = np.random.choice(self.args.memory_size, size=self.args.rl_batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.args.rl_batch_size)
        ############# train ###########################
        batch_memory = self.memory[sample_index,:]
        batch_memory = torch.tensor(batch_memory,dtype=torch.float).to(args.device)
        self.eval_net.eval()
        self.target_net.eval()

        # 输入s_到target预测下一s的val
        # q_next = self.target_net(batch_memory[:, -self.s_dim:])
        # doubel dqn的改进：同时把s_输入到eval中
        q_next_eval = self.eval_net(batch_memory[:, -self.s_dim:])

        # 输入当前s到eval预测当前各个action的val
        self.eval_net.train()
        q_eval = self.eval_net(batch_memory[:,:self.s_dim])
        q_eval_vals, _ = torch.topk(q_eval,self.args.select_node_num)
        q_eval_vals = torch.sum(q_eval_vals,dim=1)
        # 从memory中获取该s下实际采取的action以及其对应的r
        # eval_act_index = batch_memory[:, self.s_dim:self.s_dim+self.args.select_node_num].long()
        rewards = batch_memory[:, self.s_dim+self.args.select_node_num]
        # q_target = q_eval.clone()

        # 从eval中找出在s_下得分最高的action
        q_next_vals, max_act4next = torch.topk(q_next_eval,self.args.select_node_num,dim=1)
        # selected_q_next = q_next[:, max_act4next]
        q_next_vals = torch.sum(q_next_vals,dim=1)
        # 将target中选中的最高分的action替换为实际score从而与预测的结果进行对比
        # q_target[:,eval_act_index] = rewards + self.args.reward_decay*selected_q_next

        loss = self.critisen(q_eval_vals, rewards + q_next_vals*self.args.reward_decay)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        return loss.item()

    def store_transition(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.args.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_nodes(self, observation):
        observation = observation[np.newaxis, :]
        observation = torch.tensor(observation,dtype=torch.float).to(self.args.device)
        self.eval_net.eval()
        with torch.no_grad():
            actions_value = self.eval_net(observation)
            values, selected_index = torch.topk(actions_value,self.args.select_node_num)
            selected_index = selected_index.cpu().numpy()

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * torch.sum(values)
        self.q.append(self.running_q)
        # 随着模型的学习随机chose的概率降低
        if np.random.uniform() > self.epsilon:  # choosing action
            selected_index = random.sample(range(self.a_dim), self.args.select_node_num)
        return selected_index, self.running_q

    def save_model(self,episode):
        torch.save(self.eval_net.state_dict(),os.path.join(self.save_pth,f"rl_episode{episode}.pth"))

def train(args):
    kerset = [18, 30, 60, 45, 65, 50]

    RL = Double_DQN(args)
    total_steps = 1
    all_nodes_data = generate_node_data(data_pth=args.train_data_pth, node_num=args.node_num,avg_alloc=args.avg_alloc)
    test_data = generate_test_data(data_pth=args.test_data_pth)
    tp_matrix = get_energy_tp_matrix(args.node_num,args.tp_ratio)
    cp_matrix = np.array([len(node_data) * args.cp_ratio for node_data in all_nodes_data])
    ef_matrix = np.zeros(args.node_num)

    per_epi_energy = []
    per_epi_reward = []
    per_epi_rlloss = []
    for episode in range(args.episodes):
        start_time = time.time()
        energy_record = []
        loss_record = []
        reward_record = []
        observation = np.zeros(args.node_num)
        # observation = standardization(observation)
        model = CNNCifar(kerset, 10)

        for iteration in range(args.iterations):
            # user rl_model select next nodes which participate learning and center node
            selected_nodes, rq = RL.choose_nodes(observation)
            center_node, all_enery_tp = get_center_node(selected_nodes, tp_matrix)
            print(f"episode{episode},iteration{iteration} selected nodes:{selected_nodes},center node:{center_node}")
            print("#"*50+f" episode{episode},iteration{iteration} "+"#"*50)
            per_iter_loss_record = []
            per_iter_acc_record = []
            per_iter_nodes_res = []
            observation_ = observation.copy()
            # nodes train
            for i in range(args.node_num):
                if i in selected_nodes:
                    # boot all node to train
                    info,tmp_loss_record,tmp_acc_record = node_train(i,copy.deepcopy(model).to(args.device),all_nodes_data[i], test_data, args)
                    per_iter_acc_record.append(tmp_acc_record)
                    per_iter_loss_record.append(tmp_loss_record)
                    per_iter_nodes_res.append(info)
                    observation_[i] = info["weight_score"] / (cp_matrix[i]+tp_matrix[center_node,i])
                    # observation_[i+args.node_num] = cp_matrix[i] + tp_matrix[center_node,i]
                    ef_matrix[i] = info["weight_score"] / (cp_matrix[i]+tp_matrix[center_node,i])
                else:
                    per_iter_acc_record.append(0)
                    per_iter_loss_record.append(0)

            # merge all node's model data
            print("start merge all node model param")
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
            per_iter_loss_record.append(per_iter_test_loss_record)
            per_iter_acc_record.append(per_iter_test_acc_record)
            os.makedirs(args.log_dir,exist_ok=True)
            save_pth = os.path.join(args.log_dir,f"episode{episode}_iteration{iteration}_loss_record.png")
            draw_plot(save_pth,per_iter_loss_record, int(max(per_iter_loss_record))+1,"loss value")
            save_pth = os.path.join(args.log_dir,f"episode{episode}_iteration{iteration}_acc_record.png")
            draw_plot(save_pth, per_iter_acc_record, 1, "acc value")

            # compute energy_cp, energy_tp
            total_ef = np.sum(ef_matrix[selected_nodes])
            all_enery_cp = np.sum(cp_matrix[selected_nodes])

            total_enery_cost = all_enery_cp + all_enery_tp
            energy_record.append(total_enery_cost)
            print(f"total cost energy:{total_enery_cost} | all_enery_cp：{all_enery_cp} | all_enery_tp: {all_enery_tp}")
            print("ef:",(total_ef**(1-args.b_ratio))/(1-args.b_ratio))
            reward = -total_enery_cost + (total_ef**(1-args.b_ratio))/(1-args.b_ratio)
            print("reward:",reward)
            reward_record.append(reward)
            observation_ = standardization(observation_)
            RL.store_transition(observation, selected_nodes, reward, observation_)

            # save model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"SBM_iter{episode}.pth"))

            if total_steps > args.memory_size:   # learning
                loss = RL.learn()
                print(f"step {total_steps}:loss:{loss}|running q:{rq}")
                per_epi_rlloss.append(loss)
                RL.save_model(episode)


                loss_record.append(per_iter_test_loss_record)

                plt.figure()
                plt.plot(np.array(RL.q), c='b', label='q')
                plt.legend(loc='best')
                plt.ylabel('q_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig(f"rl_log/episode{episode}_DQ.png")
                # plt.show()

                plt.figure()
                plt.plot(np.array(reward_record), c='g', label='reward')
                plt.legend(loc='best')
                plt.ylabel('reward_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig(f"rl_log/episode{episode}_reward.png")
                # plt.show()

                plt.figure()
                plt.plot(np.array(energy_record), c='y', label='energy_cost')
                plt.legend(loc='best')
                plt.ylabel('energy_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig(f"rl_log/episode{episode}_E.png")
                # plt.show()

                plt.figure()
                plt.plot(np.array(loss_record), c='r', label='loss')
                plt.legend(loc='best')
                plt.ylabel('loss_value')
                plt.xlabel('iteration')
                plt.grid()
                plt.savefig(f"rl_log/episode{episode}_Loss.png")
                # plt.show()

                plt.figure()
                plt.plot(np.array(per_epi_rlloss), c='b', label='rl_loss')
                plt.legend(loc='best')
                plt.ylabel('loss_value')
                plt.xlabel('steps')
                plt.grid()
                plt.savefig("rl_log/Rl_Loss.png")

            # if total_steps - args.memory_size > 100:   # stop game
            #     break

            observation = observation_
            total_steps += 1


        per_epi_energy.append(sum(energy_record))
        per_epi_reward.append(RL.running_q)
        # reset record
        RL.q = []
        RL.running_q = 0

        plt.figure()
        plt.plot(np.array(per_epi_reward), c='g', label='q_record')
        plt.legend(loc='best')
        plt.ylabel('q_value')
        plt.xlabel('episode')
        plt.grid()
        plt.savefig("rl_log/DQ.png")
        # plt.show()

        plt.figure()
        plt.plot(np.array(per_epi_energy), c='y', label='energy_cost')
        plt.legend(loc='best')
        plt.ylabel('energy_value')
        plt.xlabel('episode')
        plt.grid()
        plt.savefig("rl_log/E.png")

        end_time = time.time()
        print(f"episode{episode}_cost time:",end_time-start_time)

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
    parser.add_argument('--select_method', type=str, default='ea',
                        help='chose the method of select nodes,ea or rl')
    parser.add_argument('--rl_weights', type=str, default='checkpoints/rl_episode3.pth',
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

def add_param(args):
    args.lr = 0.005
    args.reward_decay = 0.9
    args.e_greedy = 0.95
    args.replace_target_iter = 60
    args.memory_size = 60 # 1000
    args.rl_batch_size = 60 # 32
    args.e_greedy_increment = 0.003

    args.episodes = 8
    args.rl_checkpoints = "./checkpoints"

if __name__ == '__main__':
    args = get_args()
    add_param(args)
    train(args)