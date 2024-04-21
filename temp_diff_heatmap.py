import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from training import all_temp_diffs

time_steps = len(all_temp_diffs)
grid_y = 3
grid_x = 3

# Converting temp_diffs to 3D numpy array
temp_diffs = np.array(all_temp_diffs).reshape(time_steps, grid_y, grid_x)

# Create a figure and axis for plotting
fig, ax = plt.subplots()
# Display the initial state of the temp_diffs grid
cax = ax.matshow(temp_diffs[0], cmap='coolwarm')
# Adding a colorbar to the plot
fig.colorbar(cax)

def update(frame):
    """Function to update the plot for each frame of the animation."""
    cax.set_data(temp_diffs[frame])
    ax.set_title(f'Temperature Difference (Step: {frame})')
    return cax,

# Create Animation
ani = FuncAnimation(fig, update, frames=range(time_steps), blit=False, interval=200)

plt.show()


