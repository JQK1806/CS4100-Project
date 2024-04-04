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

def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    loss = 0
    for log_prob, reward in zip(log_probs, rewards):
        loss += -torch.sum(log_prob) * reward  # Use the reward for each step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def concatenate_inputs(state, temp_differences):
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
    temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
    concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
    return concatenated_input

max_steps_per_ep = 24
num_episodes = 1000
energy_cost = 2.0
def q_learning():
    total_loss = 0.0
    epsilon = .9
    all_rewards = []
    all_losses = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        # generate target temps
        target_temps = np.random.randint(60, 81, size=9)
        outside_temps = generate_outside_temperatures()
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)

        log_probs = []
        rewards = []

        # set all current temps to be the ouside temp
        for hour in range (max_steps_per_ep):
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            print("State", state)
            print("Temp diffs", temp_differences)
            curr_concatenated_input = concatenate_inputs(state, temp_differences)  # Concatenate along the columns (second dimension)
            if np.random.random() < epsilon: #epsilon greedy
                actions = torch.randint(0, 4, (9,), dtype=torch.long)  # Generate random actions for 9 zones
                prob_action = torch.full((9,), 0.25)  # Each action has a probability of 0.25
                log_prob = torch.log(prob_action)  # Log probability of each action

            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
                temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
                concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
                actions, probabilities = net(concatenated_input)
                log_prob = torch.log(probabilities)  # Log probability of the action taken
            # add temp diff to input param
            # get differences
            # pass as input to the neural network with the occupancies
            # get actions
            # pass actions to environment
            # get rewards
            # and new state
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)
            
            current_temps = next_curr_temps
            temp_differences = target_temps - current_temps
            episode_reward += reward
            rewards.append(reward)
            log_probs.append(log_prob)
            next_temp_differences = target_temps - current_temps
            state = next_state
            current_temps = next_curr_temps
            temp_differences = next_temp_differences
            episode_reward += reward

            epsilon = epsilon * 0.9
        
        print(f"Episode reward: {episode_reward}")
        print("log probs", log_probs)
        print("Rewards", rewards)

        total_loss += update_policy(rewards, log_probs, optimizer)
        print(f"Training loss: {total_loss}")
        all_rewards.append(episode_reward)
        all_losses.append(total_loss)
    print('Finished Training')
    print("All rewards", all_rewards)
    print("All losses", all_losses)


q_learning()
