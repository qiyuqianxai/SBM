import torch
from torchsummary import summary
from RL_train import DQN_Net

def get_model(kerset):
    nodes = [500, 300, 200, 5]
    model = CurModel(nodes, kerset)
    compression_ratio = calculate_ratio(nodes,kerset)
    print(f"model param compression ratio: {compression_ratio:.2f}")
    return model

if __name__ == '__main__':
    # kerset = [3, 3, 2, 2],
    # model = get_model(kerset=kerset)
    # model.load_state_dict(torch.load("checkpoints/",map_location=lambda storage, loc: storage))
    # model.eval()
    model = DQN_Net(20,20)
    model.to("cuda")
    summary(model,(1,20))

