import gym
from gym import spaces
import numpy as np
import itertools

class SmartHomeEnv(gym.Env):
    def __init__(self):
        super(SmartHomeEnv, self).__init__()
        self.num_zones = 9
        self.observation_space = spaces.Discrete(2 ** self.num_zones)  # Number of possible states, 2^9 - 0=empty, 1=occupied
        self.action_space = spaces.MultiDiscrete([4] * self.num_zones)  # Each zone has 4 possible actions:0,1,2,3
        self.states = list(itertools.product([0, 1], repeat=self.num_zones))  # All possible states, (zone #, occupancy)
        self.state = self.reset()


    def reset(self):
        self.state = np.zeros(self.num_zones, dtype=int)  # Reset all zones to unoccupied - 0

    def step(self, actions, outside_temp, energy_cost, current_temps, target_temps):
        # ppl move around, current temp of each zone should be updated
        updated_temps = current_temps
        action_temp_penalty = {0:0, 1:-2, 2: -5, 3:-10}
        rewards = []
        for i, action in enumerate(actions):
            current_temps[i] = action_temp_penalty[action] + 0.5 * (outside_temp - current_temps[i])
            rewards[i] = -1 * action * energy_cost + 10 * (current_temps[i] - target_temps[i])
        
        
        self.state = np.random.randint(2, size=self.num_zones)

        return current_temps, rewards, self.state


        





        
