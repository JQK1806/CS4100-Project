import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from environment import SmartHomeEnv
from neural_network import CombinedNetwork

# Create an instance of the environment
env = SmartHomeEnv()

# Create an instance of the neural network
net = CombinedNetwork()

'''PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

'''
PART 5:
Train your model!
'''


max_steps_per_ep = 24
num_episodes = 100
def q_learning():
    for episode in range(num_episodes):
        total_loss = 0.0
        state = env.reset()
        episode_reward = 0.0
        temp_diffs = np.zeros(9)
        outside_temp = 70
        for hour in range (max_steps_per_ep):
            # add temp diff to input param
            actions = net(torch.tensor(state,dtype=torch.float32))
            next_state, rewards, done = env.step(actions)
            episode_reward += sum(rewards)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            total_loss += criterion.item()
            state = next_state

        print(f"Training loss: {total_loss}")

    print('Finished Training')


