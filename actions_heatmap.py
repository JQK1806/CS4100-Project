import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from training import all_actions 
from matplotlib.colors import Normalize


time_steps = len(all_actions)
grid_y = 3
grid_x = 3

# Convert actions into 3D numpy array
actions_array = np.array(all_actions).reshape(time_steps, grid_y, grid_x)

# Create a figure and axis for plotting
fig, ax = plt.subplots()

# Display the initial state of the actions grid
cax = ax.matshow(actions_array[0], cmap='coolwarm')

action_labels = ['Off', 'Low', 'Medium', 'High']

# Normalize colors based on the number of action levels
norm = Normalize(vmin=0, vmax=len(action_labels)-1)

# Generate colors corresponding to action labels
colors = plt.cm.coolwarm(norm(range(len(action_labels))))


def update(frame):
    """Function to update the plot for each frame of the animation."""
    cax.set_data(actions_array[frame]) 
    ax.set_title(f'Actions (Step: {frame})')
    return cax,

# Create Animation
ani = FuncAnimation(fig, update, frames=range(time_steps), blit=False, interval=200)

# Add legend
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(action_labels))],
           labels=action_labels,
           loc='center left',
           bbox_to_anchor=(1, 0.5))

plt.show()
