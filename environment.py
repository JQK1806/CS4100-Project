import gym
from gym import spaces
import numpy as np

class SmartHomeEnv(gym.env):
    def __init__(self, params):
        super(SmartHomeEnv, self).__init__()
        self.params = params
        self.num_appliances = params['num_appliances']
        self.appliances = params['appliances']
        self.appliance_energy_consumption = [appliance['energy_consumption'] for appliance in self.appliances]

        self.action_space = spaces.MultiBinary(self.num_appliances)  # On/Off for each appliance
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_appliances,), dtype=np.float32)  # Status of appliances
        self.state = self.reset()
    def reset(self):
        pass
    def step(self, action):
        pass
    


