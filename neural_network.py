import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniZoneNetwork(nn.Module):
    def __init__(self):
        super(MiniZoneNetwork, self).__init__()
        self.fc1 = nn.Linear(2,3) # 2 inputs (occupancy, temp diff), 3 hidden units
        self.fc2 = nn.Linear(3,4) # 3 hidden units, 4 outputs (0,1,2,3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()
        self.zones = nn.ModuleList([MiniZoneNetwork() for _ in range(9)])
    def forward(self, inputs):
        outputs=[]
        for i in range(9):
            # used chat gpt
            output_i = self.zones[i](inputs[:, i, :])  # pass ith zone's input to its mini network
            output_i = F.softmax(output_i, dim=-1)  #  softmax to get probabilities of each action 
            max_action_i = torch.argmax(output_i, dim=-1)  # Choose the action with the highest probability
            outputs.append(max_action_i)
        return torch.stack(outputs, dim=1)  # stack outputs of all mini networks along dimension 1 - output is (9,2)
