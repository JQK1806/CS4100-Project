import gym
from gym import spaces
import numpy as np
import torch 

class SmartHomeEnv(gym.Env):
    def __init__(self):
        super(SmartHomeEnv, self).__init__()
        self.num_zones = 9
        self.observation_space = spaces.Discrete(2 ** self.num_zones)  # Number of possible states, 2^9 - 0=empty, 1=occupied
        self.action_space = spaces.MultiDiscrete([4] * self.num_zones)  # Each zone has 4 possible actions:0,1,2,3
        #self.states = list(itertools.product([0, 1], repeat=self.num_zones))  # All possible states, (zone #, occupancy)
        self.state = self.reset()


    def reset(self):
        self.state = np.zeros(self.num_zones, dtype=int)  # Reset all zones to unoccupied - 0
        return self.state

    def step(self, actions, outside_temp, energy_cost, current_temps, target_temps):
        actions = actions.numpy()
        # ppl move around, current temp of each zone should be updated
        action_temp_penalty = {0:0, 1:-4, 2: -8, 3:-12}
        reward = 0
        for i, action in enumerate(actions):
            print(f"Zone: {i}, Action: {action}")
            print(f"Current Temp: {current_temps[i]}")
            print(f"Target Temp: {target_temps[i]}")
            print(f"Outside Temp: {outside_temp}")
            print("Natural temp change:",  0.2 * (outside_temp - current_temps[i]))
            nat_temp_change = 0.2 * (outside_temp - current_temps[i])

            if target_temps[i] > current_temps[i]:
                 action_temp_change = -action_temp_penalty[action]
            else:
                action_temp_change = action_temp_penalty[action]
            current_temps[i] = current_temps[i] + nat_temp_change + action_temp_change
            print(f"New Temp: {current_temps[i]}")
            temp_diff = current_temps[i] - target_temps[i]
            action_reward = action * -energy_cost
            temp_reward = -20 * (abs(temp_diff) - (self.state[i]-1)) # take into account occupancy
            print(f"Reward: {action_reward + temp_reward}")
            reward = action_reward + temp_reward + reward



        self.state = np.random.randint(2, size=self.num_zones)

        return self.state, current_temps, reward


        





        
