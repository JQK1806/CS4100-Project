import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from training import all_temp_diffs

time_steps = len(all_temp_diffs)
grid_y = 3
grid_x = 3

temp_diffs = np.array(all_temp_diffs).reshape(time_steps, grid_y, grid_x)

fig, ax = plt.subplots()
cax = ax.matshow(temp_diffs[0], cmap='coolwarm')
fig.colorbar(cax)

def update(frame):
    cax.set_data(temp_diffs[frame])
    ax.set_title(f'Temperature Difference (Episode: {frame})')
    return cax,

ani = FuncAnimation(fig, update, frames=range(time_steps), blit=False, interval=200)

plt.show()