import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from training import all_actions 
from matplotlib.colors import Normalize


time_steps = len(all_actions)
grid_y = 3
grid_x = 3

actions_array = np.array(all_actions).reshape(time_steps, grid_y, grid_x)

fig, ax = plt.subplots()
cax = ax.matshow(actions_array[0], cmap='coolwarm')

action_labels = ['Off', 'Low', 'Medium', 'High']
norm = Normalize(vmin=0, vmax=len(action_labels)-1)

colors = plt.cm.coolwarm(norm(range(len(action_labels))))


def update(frame):
    cax.set_data(actions_array[frame]) 
    ax.set_title(f'Actions (Episode: {frame})')
    return cax,

ani = FuncAnimation(fig, update, frames=range(time_steps), blit=False, interval=200)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(action_labels))],
           labels=action_labels,
           loc='center left',
           bbox_to_anchor=(1, 0.5))

plt.show()
