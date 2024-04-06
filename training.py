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


def calculate_loss(reward, log_probs):
    loss = 0
    for log_prob in log_probs:
        loss += log_prob * reward  # Use the reward for each step
    return loss

def concatenate_inputs(state, temp_differences):
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
    temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
    concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
    return concatenated_input



max_steps_per_ep = 24
num_episodes = 1000
energy_cost = 5.0

def generate_target_temps(num_episodes):
    target_temps_all_episodes = np.random.randint(60, 81, size=(num_episodes, 9))
    return target_temps_all_episodes

def generate_outside_temperatures(num_episodes):
    return np.random.randint(60, 81, size=(num_episodes, 24))
target_temps_for_all_episodes = generate_target_temps(num_episodes)
outside_temps_for_all_episodes = generate_outside_temperatures(num_episodes)

def training():
    all_temp_diffs = []
    all_actions = []
    epsilon = .99
    all_rewards = []
    all_losses = []
    for episode in range(num_episodes):
        ep_loss = 0.0
        state = env.reset()
        episode_reward = 0.0
        # generate target temps
        target_temps = target_temps_for_all_episodes[episode]
        outside_temps = outside_temps_for_all_episodes[episode]
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)
        # set all current temps to be the ouside temp
        for hour in range (max_steps_per_ep):
            if hour >= 24: 
                hour = hour % 24 
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            all_temp_diffs.append(temp_differences)

            curr_concatenated_input = concatenate_inputs(state, temp_differences)  # Concatenate along the columns (second dimension)
            if np.random.random() < epsilon: #epsilon greedy
                actions = torch.randint(0, 4, (9,), dtype=torch.long)  # Generate random actions for 9 zones
                prob_action = torch.full((9,), 0.25)  # Each action has a probability of 0.25
                log_probs = torch.log(prob_action)  # Log probability of each action
            else:
                actions, probabilities = net(curr_concatenated_input)
                log_probs = torch.log(probabilities)  # Log probability of the action taken
                # print(f"Probabilities: {probabilities}")
            # add temp diff to input param
            # get differences
            # pass as input to the neural network with the occupancies
            # get actions
            # pass actions to environment
            # get rewards
            # and new state
            # print(f"Hour: {hour}")
            # print(f"State: {state}")
            # print(f"Current temps: {current_temps}")
            # print(f"Target temps: {target_temps}")
            # print(f"Outside temp: {outside_temp}")
            # print(f"Actions: {actions}")
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)

            epsilon = epsilon * 0.995
            ep_loss += calculate_loss(reward, log_probs)

            current_temps = next_curr_temps
            temp_differences = target_temps - current_temps
            episode_reward += reward
            state = next_state
            # print(f"Next state: {state}")
            # print(f"Updated current temps after action: {current_temps}")
            print(f"Reward: {reward}")
            # print(f"Step loss: {calculate_loss(reward, log_probs)}")

        
        # print(f"Episode reward: {episode_reward}")
        print(f"Training loss: {ep_loss.item()}")

         # Perform the gradient computation and optimization
        optimizer.zero_grad()  # Reset gradients
        ep_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        all_rewards.append(episode_reward)
        all_losses.append(ep_loss.item())
        all_actions.append(actions.numpy())

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
    # print("Last reward", all_rewards[999])
    # print("Last loss", all_losses[999])
    return all_rewards, all_losses, all_temp_diffs, all_actions

rewards, losses, all_temp_diffs, all_actions = training()
plt.figure()
plt.plot(rewards, label='Rewards', color='#396336')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Over Episodes For Our Model')
plt.legend()
plt.tight_layout()  # Adjust plot to prevent labels from being cut off
plt.savefig('model_rewards_plot.png')  # Save the plot as a PNG image
plt.close()

plt.figure()
plt.plot(losses, label='Loss', color='#396336')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss Over Episodes For Our Model')
plt.legend()
plt.tight_layout()  # Adjust plot to prevent labels from being cut off
plt.savefig('loss_plot.png')  # Save the plot as a PNG image
plt.close()


def random_model():
    all_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        # generate target temps
        target_temps = target_temps_for_all_episodes[episode]
        outside_temps = outside_temps_for_all_episodes[episode]
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)
        # set all current temps to be the ouside temp
        for hour in range (max_steps_per_ep):
            if hour >= 24: 
                hour = hour % 24 
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            all_temp_diffs.append(temp_differences)
            actions = torch.randint(0, 4, (9,), dtype=torch.long)  # Generate random actions for 9 zones
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)
            current_temps = next_curr_temps
            temp_differences = target_temps - current_temps
            episode_reward += reward
            state = next_state
        all_rewards.append(episode_reward)
    print('Finished Training')
    print("All rewards", all_rewards)
    # print("Last reward", all_rewards[999])
    # print("Last loss", all_losses[999])
    return all_rewards

random_rewards = random_model()
plt.figure()
plt.plot(random_rewards, label='Random Rewards', color='#396336')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Over Episodes for Random Choices')
plt.legend()
plt.tight_layout()  # Adjust plot to prevent labels from being cut off
plt.savefig('random_model_rewards_plot.png')  # Save the plot as a PNG image
plt.close()

