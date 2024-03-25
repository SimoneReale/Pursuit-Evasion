import matplotlib.pyplot as plt
from utils import n_of_nodes, n_time, n_preys
import pickle

# Number of nodes
nodes = [36, 49, 64, 81]

with open(f"mver_{10}_n{36}_p{2}_t{20}.pkl", "rb") as fp: 
    m1_36 = pickle.load(fp)

with open(f"mver_{10}_n{49}_p{2}_t{24}.pkl", "rb") as fp: 
    m1_49 = pickle.load(fp)

with open(f"mver_{10}_n{64}_p{2}_t{28}.pkl", "rb") as fp: 
    m1_64 = pickle.load(fp)

with open(f"mver_{10}_n{81}_p{2}_t{32}.pkl", "rb") as fp: 
    m1_81 = pickle.load(fp)

with open(f"mver_{10}_n{100}_p{2}_t{36}.pkl", "rb") as fp: 
    m1_100 = pickle.load(fp)


with open(f"MSimple_mver_{10}_n{36}_p{2}_t{20}.pkl", "rb") as fp: 
    m2_36 = pickle.load(fp)

with open(f"MSimple_mver_{10}_n{49}_p{2}_t{24}.pkl", "rb") as fp: 
    m2_49 = pickle.load(fp)

with open(f"MSimple_mver_{10}_n{64}_p{2}_t{28}.pkl", "rb") as fp: 
    m2_64 = pickle.load(fp)

with open(f"MSimple_mver_{10}_n{81}_p{2}_t{32}.pkl", "rb") as fp: 
    m2_81 = pickle.load(fp)

with open(f"MSimple_mver_{10}_n{100}_p{2}_t{36}.pkl", "rb") as fp: 
    m2_100 = pickle.load(fp)

mean_percentages_advanced = []
for x in [m1_36, m1_49, m1_64, m1_81]:
    total_perc = 0
    for y in x:
        total = y["total_n_of_samples"]
        feasible = y["n_of_feasible"]
        total_perc += round(100 * (feasible / total), 2)
    mean_percentages_advanced.append(round(total_perc / len(x), 2))


mean_percentages_simple = []
for x in [m2_36, m2_49, m2_64, m2_81]:
    total_perc = 0
    for y in x:
        total = y["total_n_of_samples"]
        feasible = y["n_of_feasible"]
        total_perc += round(100 * (feasible / total), 2)
    mean_percentages_simple.append(round(total_perc / len(x), 2))


plt.rcParams.update({'font.size': 28})

# Feasibility percentage for each model (example data)
model1_feasibility = mean_percentages_advanced  # Feasibility percentage for Model 1
model2_feasibility = mean_percentages_simple  # Feasibility percentage for Model 2

# Calculate overlap for better visualization
overlap = 0.3

# Define bar width
bar_width = 0.35

# Calculate position for the bars
bar1_position = [x - overlap/2 for x in range(len(nodes))]
bar2_position = [x + overlap/2 for x in range(len(nodes))]

# Create bar chart
plt.bar(bar1_position, model1_feasibility, width=bar_width, label='Model Advanced')
plt.bar(bar2_position, model2_feasibility, width=bar_width, label='Model Simple')

# Set labels and title
plt.xlabel('Number of Nodes')
plt.ylabel('Mean Feasibility % of the samples')
#plt.title('Feasibility Percentage of Models by Number of Nodes')
plt.xticks([r for r in range(len(nodes))], nodes)

# Add legend
plt.legend()

# Show plot
plt.show()
