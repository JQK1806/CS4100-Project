import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniZoneNetwork(nn.Module):
    def __init__(self):
        super(MiniZoneNetwork, self).__init__()
        self.fc1 = nn.Linear(2,3) # 2 inputs (occupancy, temp diff), 3 hidden units
        self.fc2 = nn.Linear(3,4) # 3 hidden units, 4 outputs (0,1,2,3)

    def forward(self, x):
        print("x", x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()
        self.zones = nn.ModuleList([MiniZoneNetwork() for _ in range(9)])
    def forward(self, inputs):
        actions=[]
        q_values = []
        for i in range(9):
            # used chat gpt
            print("self zones", self.zones)
            print("inputs", inputs)
            print("mini zone input i", inputs[:,i])
            output_i = self.zones[i](inputs[:, i])  # pass ith zone's input to its mini network
            q_values.append(output_i)
            output_i = F.softmax(output_i, dim=-1)  #  softmax to get probabilities of each action 
            max_action_i = torch.argmax(output_i, dim=-1)  # Choose the action with the highest probability
            print("max action i", max_action_i)
            actions.append(max_action_i)
        print("outputs", torch.stack(actions, dim=0))
        print("q values", torch.stack(q_values, dim=0))
        return torch.stack(actions, dim=0), torch.stack(q_values, dim=0) # stack outputs of all mini networks along dim 1
