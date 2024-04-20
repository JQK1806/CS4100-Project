from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
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


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def calculate_loss(reward, log_probs):
    loss = 0
    
    # Calculate loss as the sum of log probabilities weighted by rewards
    for log_prob in log_probs:
        loss += log_prob * reward  # Use the reward for each step
    return loss

def concatenate_inputs(state, temp_differences):
    # used Chat GPT
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)  # Reshape next_state to a row vector
    temp_differences_tensor = torch.tensor(temp_differences, dtype=torch.float32).reshape(1, -1)  # Reshape temp_differences to a row vector
    concatenated_input = torch.cat((state_tensor, temp_differences_tensor), dim=0)  # Concatenate along the columns (second dimension)
    return concatenated_input

# Set parameters for training
max_steps_per_ep = 24
num_episodes = 1000
energy_cost = 5.0


def generate_target_temps(num_episodes):
    # Generate random target temps for each zone to simulate varying user preferences
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
        
        # Generate target temps, outside temps
        target_temps = target_temps_for_all_episodes[episode]
        outside_temps = outside_temps_for_all_episodes[episode]
        outside_temp = outside_temps[0]
        current_temps = np.full(9, outside_temp)
        
        # Simulating each hour within an episode
        for hour in range (max_steps_per_ep):
            if hour >= 24: 
                hour = hour % 24 
            outside_temp = outside_temps[hour]
            temp_differences = target_temps - current_temps
            all_temp_diffs.append(temp_differences)
            curr_concatenated_input = concatenate_inputs(state, temp_differences)  # Concatenate along the columns (second dimension)
            
            if np.random.random() < epsilon: #Epsilon greedy strategy
                actions = torch.randint(0, 4, (9,), dtype=torch.long)  # Generate random actions for 9 zones
                prob_action = torch.full((9,), 0.25)  # Each action has a probability of 0.25
                log_probs = torch.log(prob_action)  # Log probability of each action
            else:
                actions, probabilities = net(curr_concatenated_input)
                log_probs = torch.log(probabilities)  # Log probability of the action taken
                
            next_state, next_curr_temps, reward = env.step(actions, outside_temp, energy_cost, current_temps, target_temps)

            # Decay epsilon value for epsilon-greedy exploration
            epsilon = epsilon * 0.995

            # Calculate episode loss
            ep_loss += calculate_loss(reward, log_probs)

            # Update current temps and calculate temp differences
            current_temps = next_curr_temps
            temp_differences = target_temps - current_temps
            
            # Accumulate episode reward
            episode_reward += reward
            
            # Update current state
            state = next_state
            print(f"Reward: {reward}")

        print(f"Training loss: {ep_loss.item()}")

        # Perform gradient computation and optimization
        optimizer.zero_grad()  # Reset gradients
        ep_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        all_rewards.append(episode_reward)
        all_losses.append(ep_loss.item())
        all_actions.append(actions.numpy())

    print('Finished Training')
    print("All rewards", all_rewards)
    print("All losses", all_losses)
    return all_rewards, all_losses, all_temp_diffs, all_actions

# Run training
rewards, losses, all_temp_diffs, all_actions = training()

# Plot rewards
plt.figure()
plt.plot(rewards, label='Rewards', color='#396336')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Over Episodes For Our Model')
plt.legend()
plt.tight_layout()  # Adjust plot to prevent labels from being cut off
plt.savefig('model_rewards_plot.png')  # Save the plot as a PNG image
plt.close()

# Plot losses
plt.figure()
plt.plot(losses, label='Loss', color='#396336')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss Over Episodes For Our Model')
plt.legend()
plt.tight_layout()  # Adjust plot to prevent labels from being cut off
plt.savefig('loss_plot.png')  # Save the plot as a PNG image
plt.close()


# Function to run random model for comparison
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



time_steps_temp = len(all_temp_diffs)
grid_y_temp = 3
grid_x_temp = 3

temp_diffs = np.array(all_temp_diffs).reshape(time_steps_temp, grid_y_temp, grid_x_temp)

fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
fig.subplots_adjust(wspace=0.5)

# Temperature subplot
cax_temp = axs[0].matshow(temp_diffs[0], cmap='coolwarm')
fig.colorbar(cax_temp, ax=axs[0])

def update_temp(frame):
    cax_temp.set_data(temp_diffs[frame])
    axs[0].set_title(f'Temperature Difference (Step: {frame})')
    return cax_temp,

ani_temp = FuncAnimation(fig, update_temp, frames=range(time_steps_temp), blit=False, interval=200)

# Actions subplot
time_steps_actions = len(all_actions)
grid_y_actions = 3
grid_x_actions = 3

actions_array = np.array(all_actions).reshape(time_steps_actions, grid_y_actions, grid_x_actions)

cax_actions = axs[1].matshow(actions_array[0], cmap='coolwarm')

action_labels = ['Off', 'Low', 'Medium', 'High']
norm = Normalize(vmin=0, vmax=len(action_labels)-1)

colors = plt.cm.coolwarm(norm(range(len(action_labels))))

 # used Chat GPT
def update_actions(frame):
    cax_actions.set_data(actions_array[frame])
    axs[1].set_title(f'Actions (Step: {frame})')
    return cax_actions,

ani_actions = FuncAnimation(fig, update_actions, frames=range(time_steps_actions), blit=False, interval=200)

# Legend for actions subplot
legend = axs[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(action_labels))],
                       labels=action_labels,
                       loc='center left',
                       bbox_to_anchor=(1, 0.5))

plt.show()


