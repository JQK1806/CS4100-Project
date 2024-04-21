# Optimizing Energy Consumption in a Smart Home Temperature System 
Our project aims to create a model reduce the cost of electricity and optimize the users’ comfort in a smart home. The model will manage the temperature of a room based on the current outside temperature and the occupancy level of the room. The training of the model will be done by utilizing reinforcement learning, which allows for the model to be well-suited for efficiently adapting to changing environments.

Our reinforcement learning smart home environment was set up with 9 zones. Each zone had a state of either 0 or 1 with 0 representing the zone being unoccupied and 1 representing the zone being occupied. Each zone also stored a current temperature and a target temperature to simulate varying user preferences between the different zones. The model was run over 1000 episodes. The action space available to take at each step consisted of actions 0,1,2, and 3. The environment was set up so an action of 0 would represent turning off the temperature control system. An action of 1 corresponds to low or a small change towards the temperature, an action of 2 corresponds to medium or a medium change towards the temperature, and an action of 3 corresponds to high or a high change towards the temperature. At each step, the current temperature for each zone was updated based on the action taken and based on the natural temperature change which was calculated based on the outside temperature. The energy cost was set to be 5 which was factored into the rewards equation.

# Setup Guide 
In order to run the model, you need to run the training.py fil 


