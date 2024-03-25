import numpy as np

N = 5  # number of nodes
T = 3  # number of time steps

# Example: Initialize the tensor with random costs. In practice, this will be your actual c_{i,j,t} values.
c_ijt = np.random.rand(T, N, N)  # Tensor with dimensions T x N x N


import matplotlib.pyplot as plt

# Create subplots: one for each time step
fig, axes = plt.subplots(1, T, figsize=(15, 5))

for t in range(T):
    ax = axes[t]
    c_matrix = c_ijt[t, :, :]
    im = ax.imshow(c_matrix, cmap="viridis", interpolation="nearest")
    
    # Labeling
    ax.set_title(f"Time {t}")
    ax.set_xlabel("Node j")
    ax.set_ylabel("Node i")
    
    # Optional: Add a color bar to indicate costs
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
