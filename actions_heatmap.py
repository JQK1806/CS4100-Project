import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from training import all_actions 

time_steps = len(all_actions)
grid_y = 3
grid_x = 3

actions_array = np.array(all_actions).reshape(time_steps, grid_y, grid_x)

fig, ax = plt.subplots()
cax = ax.matshow(actions_array[0], cmap='coolwarm')
fig.colorbar(cax)

def update(frame):
    cax.set_data(actions_array[frame]) 
    ax.set_title(f'Actions (Episode: {frame})')
    return cax,

ani = FuncAnimation(fig, update, frames=range(time_steps), blit=False, interval=200)

plt.show()
