import numpy as np
import matplotlib.pyplot as plt

# Example data: solution quality for Model 1 and Model 2
model1_solution_quality = np.random.normal(loc=100, scale=20, size=1000)
model2_solution_quality = np.random.normal(loc=110, scale=25, size=1000)

# Create histogram
plt.figure(figsize=(8, 6))
plt.hist(model1_solution_quality, bins=30, alpha=0.5, label='Model 1', color='blue')
plt.hist(model2_solution_quality, bins=30, alpha=0.5, label='Model 2', color='orange')
plt.xlabel('Solution Quality')
plt.ylabel('Frequency')
plt.title('Distribution of Solution Quality')
plt.legend()
plt.grid(True)
plt.show()
