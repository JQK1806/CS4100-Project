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
        # ppl move around, current temp of each zone should be updated
        action_temp_penalty = {0:0, 1:-2, 2: -5, 3:-10}
        reward = 0
        for i, action in enumerate(actions):
            if target_temps[i] > current_temps[i]:
                 current_temps[i] = -action_temp_penalty[action]
            else:
                current_temps[i] = action_temp_penalty[action]
            if current_temps[i] < outside_temp:
                current_temps[i] = current_temps[i] + 0.5 * (outside_temp - current_temps[i])
            else:
                current_temps[i] = current_temps[i] - 0.5 * (current_temps[i] - outside_temp)
            temp_diff = current_temps[i] - target_temps[i]
            action_reward = -1 * action * energy_cost
            temp_reward = 10 * (10 - abs(temp_diff) + self.state[i]) # take into account occupancy
            reward = action_reward + temp_reward + reward



        self.state = np.random.randint(2, size=self.num_zones)

        return self.state, current_temps, reward


        





        
