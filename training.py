import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from environment import SmartHomeEnv
from neural_network import CombinedNetwork
from rand_weather_data import generate_outside_temperatures
import matplotlib.pyplot as plt

# Create an instance of the environment
env = SmartHomeEnv()

# Create an instance of the neural network
net = CombinedNetwork()

'''PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

'''
PART 5:
Train your model!
'''


def calculate_loss(reward, log_prob):
    loss = 0
    for log_prob in log_prob:
        loss += log_prob * reward  # Use the reward for each step
    return loss

def concatenate_inputs(state, temp_differences):
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
    temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
    concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
    return concatenated_input

max_steps_per_ep = 300
num_episodes = 600
energy_cost = 5.0
def q_learning():
    epsilon = .99
    all_rewards = []
    all_losses = []
    for episode in range(num_episodes):
        ep_loss = 0.0
        state = env.reset()
        episode_reward = 0.0
        # generate target temps
        target_temps = np.random.randint(60, 81, size=9)
        outside_temps = generate_outside_temperatures()
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)

        log_probs = []

        # set all current temps to be the ouside temp
        for hour in range (max_steps_per_ep):
            if hour >= 24: 
                hour = hour % 24 
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            curr_concatenated_input = concatenate_inputs(state, temp_differences)  # Concatenate along the columns (second dimension)
            if np.random.random() < epsilon: #epsilon greedy
                actions = torch.randint(0, 4, (9,), dtype=torch.long)  # Generate random actions for 9 zones
                prob_action = torch.full((9,), 0.25)  # Each action has a probability of 0.25
                log_prob = torch.log(prob_action)  # Log probability of each action
            else:
                actions, probabilities = net(curr_concatenated_input)
                log_prob = torch.log(probabilities)  # Log probability of the action taken
            # add temp diff to input param
            # get differences
            # pass as input to the neural network with the occupancies
            # get actions
            # pass actions to environment
            # get rewards
            # and new state
            print(f"Hour: {hour}")
            print(f"State: {state}")
            print(f"Current temps: {current_temps}")
            print(f"Target temps: {target_temps}")
            print(f"Outside temp: {outside_temp}")
            print(f"Actions: {actions}")
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)
            log_probs.append(log_prob)

            epsilon = epsilon * 0.99
            ep_loss += calculate_loss(reward, log_prob)

            current_temps = next_curr_temps
            temp_differences = target_temps - current_temps
            episode_reward += reward
            state = next_state
            print(f"Next state: {state}")
            print(f"Updated current temps after action: {current_temps}")
            print(f"Reward: {reward}")
            print(f"Step loss: {calculate_loss(reward, log_prob)}")

        
        print(f"Episode reward: {episode_reward}")
        print(f"Training loss: {ep_loss.item()}")

         # Perform the gradient computation and optimization
        optimizer.zero_grad()  # Reset gradients
        ep_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        all_rewards.append(episode_reward)
        all_losses.append(ep_loss.item())

        # # Plot the training loss after every episode
        # plt.plot(all_losses, label='Training Loss')
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Over Episodes')
        # plt.legend()
        # plt.show()

        # # Plot the rewards after every episode
        # plt.plot(all_rewards, label='Training Loss')
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.title('Rewards Over Episodes')
        # plt.legend()
        # plt.show()
    print('Finished Training')
    print("All rewards", all_rewards)
    print("All losses", all_losses)
    return all_rewards, all_losses



rewards, losses = q_learning()
plt.plot(rewards, label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Over Episodes')
plt.legend()
plt.show()

plt.plot(losses, label='Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss Over Episodes')
plt.legend()
plt.show()
