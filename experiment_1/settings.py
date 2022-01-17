import math
import random
import numpy as np

random.seed(6)

def get_energy_tp_matrix(node_num=10,ratio=1):
    node_pos_matrix = list(range(100))

    res = random.sample(node_pos_matrix, node_num)

    for i in range(100):
        if i in res:
            node_pos_matrix[i] = 1
        else:
            node_pos_matrix[i] = 0

    node_pos_matrix = np.array(node_pos_matrix).reshape(10,10)
    print("node_position:\n",node_pos_matrix)
    positions = np.where(node_pos_matrix==1)
    positions = [[x,y] for x,y in zip(positions[0],positions[1])]
    tp_matrix = []
    for i,pos_1 in enumerate(positions):
        tmp = []
        for j,pos_2 in enumerate(positions):
            dist_tp = math.sqrt((pos_1[0]-pos_2[0])**2+(pos_1[1]-pos_2[1])**2)*ratio
            tmp.append(dist_tp)
        tp_matrix.append(tmp)
    tp_matrix = np.array(tp_matrix)
    print("tp_energy matrix:\n",tp_matrix)
    return tp_matrix

def get_center_node(node_ids,tp_matrix):
    nodes = sorted(node_ids)
    # print(nodes)
    # center_nodes = nodes[int(len(nodes)/2)-1:int(len(nodes)/2)+1]
    # print(center_nodes)
    # node_1_e = np.sum(tp_matrix[center_nodes[0],nodes])
    # node_2_e = np.sum(tp_matrix[center_nodes[1],nodes])
    all_nodes_energy = []
    for node in nodes:
        e = np.sum(tp_matrix[node,nodes])
        # print(node,"tp energy:",e)
        all_nodes_energy.append(e)
    center_node = np.argmin(all_nodes_energy)

    return nodes[center_node],all_nodes_energy[center_node]




if __name__ == '__main__':
    tp_matrix = get_energy_tp_matrix(node_num=15,ratio=1)


    node_num = 15
    selected_node_num = 10
    selected_node_id = random.sample(list(range(node_num)), selected_node_num)
    center_node,e = get_center_node(selected_node_id,tp_matrix)
    print("center node:",center_node,e)





