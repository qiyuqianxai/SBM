import torch
import argparse
from SBM_train import SBM
from utils import generate_node_data,generate_test_data
from nets import CNNCifar
from settings import get_energy_tp_matrix
# get all nodes data to allocate

torch.cuda.set_device(1)
def get_all_nodes_data(data_pth, node_num,avg_alloc):
    all_node_data = generate_node_data(data_pth,node_num,avg_alloc)
    return all_node_data

# get Rolser model
def get_model(kerset,num_classes):
    model = CNNCifar(kerset,num_classes)
    return model

def main(args):
    model = get_model(args.kerset,args.num_classes)
    if args.pretrained_weight:
        model.load_state_dict(torch.load(args.pretrained_weight,map_location=lambda storage, loc:storage))
    all_nodes_data = generate_node_data(data_pth = args.train_data_pth, node_num = args.node_num, random_imgs = args.random_imgs, avg_alloc = args.avg_alloc)
    test_data = generate_test_data(data_pth = args.test_data_pth)

    SBM(model, all_nodes_data, test_data, args)

    for arg in vars(args):
        print(arg, getattr(args, arg))

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
    parser.add_argument('--select_method', type=str, default=None,
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

if __name__ == '__main__':
    args = get_args()
    main(args)

