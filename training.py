import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from environment import SmartHomeEnv
from neural_network import CombinedNetwork
from rand_weather_data import generate_outside_temperatures

# Create an instance of the environment
env = SmartHomeEnv()

# Create an instance of the neural network
net = CombinedNetwork()

'''PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
discount = .9

'''
PART 5:
Train your model!
'''

def concatenate_inputs(state, temp_differences):
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
    temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
    concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
    return concatenated_input

max_steps_per_ep = 24
num_episodes = 100

energy_cost = 2.0
def q_learning():
    total_loss = 0.0
    epsilon = .9
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        # generate target temps
        target_temps = np.random.randint(60, 81, size=9)
        outside_temps = generate_outside_temperatures()
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)
        # set all current temps to be the ouside temp
        for hour in range (max_steps_per_ep):
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            print("State", state)
            print("Temp diffs", temp_differences)
            curr_concatenated_input = concatenate_inputs(state, temp_differences)  # Concatenate along the columns (second dimension)
            if np.random.random() < epsilon: #epsilon greedy
                actions = np.random.randint(4, size=9)
            else:
                actions = net(curr_concatenated_input)
            # add temp diff to input param
            # get differences
            # pass as input to the neural network with the occupancies
            # get actions
            # pass actions to environment
            # get rewards
            # and new state
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)
            print("Reward", reward)
            print("Next curr temps", next_curr_temps)
            print("target temps", target_temps)
            next_temp_differences = target_temps - current_temps
            print("Temp diffs", next_temp_differences)

            next_concatenated_input = concatenate_inputs(next_state, next_temp_differences)  # Concatenate along the columns (second dimension)
            print("CONCATENATED INPUT", next_concatenated_input)
            
            # Calculate the target Q-value for the next state
            _, next_q_values = net(next_concatenated_input)
            max_next_q_value = next_q_values.max().item()
            target_q_value = reward + discount * max_next_q_value if hour != 23 else reward

            # Get Q-values for the current state
            _, q_values = net(curr_concatenated_input)
            
            # Convert actions to a tensor
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            
            # Get Q-value for the current state and action
            current_q_value = torch.gather(q_values, 1, actions_tensor.unsqueeze(1))
            print("Current Q value", current_q_value)

             # Calculate the loss
            loss = criterion(current_q_value, torch.tensor(target_q_value, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the total loss and current state
            total_loss += loss.item()
            state = next_state
            current_temps = next_curr_temps
            temp_differences = next_temp_differences
            episode_reward += reward

            epsilon = epsilon * 0.9

        print(f"Training loss: {total_loss}")

    print('Finished Training')


q_learning()
