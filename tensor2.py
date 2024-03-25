import numpy as np
import matplotlib.pyplot as plt

N = 8  
T = 4  
min_cost = 5  
max_cost = 15  

plt.rcParams.update({'font.size': 20})

c_ijt = np.random.randint(min_cost, max_cost+1, size=(T, N, N))

fig, axes = plt.subplots(1, T, figsize=(15, 5), sharex=True, sharey=True)


node_ticks = np.arange(N)
tick_labels = [f"{i}" for i in range(N)]

for t in range(T):
    ax = axes[t]
    c_matrix = c_ijt[t, :, :]
    im = ax.imshow(c_matrix, cmap="viridis", interpolation="nearest", vmin=min_cost, vmax=max_cost)
    
    ax.set_title(f"Time {t}")
    ax.set_xlabel("Node j")
    ax.set_ylabel("Node i")
    

    ax.set_xticks(node_ticks)
    ax.set_xticklabels(tick_labels, ha="right")
    ax.set_yticks(node_ticks)
    ax.set_yticklabels(tick_labels)

plt.subplots_adjust(bottom=0.35)


cbar_ax = fig.add_axes([0.15, 0.3, 0.7, 0.05])


cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

cbar.set_ticks(np.linspace(min_cost, max_cost, num=max_cost-min_cost+1))
cbar.set_ticklabels(np.arange(min_cost, max_cost+1))
cbar_ax.set_xlabel('Cost')

plt.show()
