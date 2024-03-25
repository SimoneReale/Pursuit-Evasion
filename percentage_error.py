import numpy as np
import matplotlib.pyplot as plt
import pickle



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





mean_percentages_simple_1 = []
for x in [m1_36, m1_49, m1_64, m1_81]:
    costs_array = []
    for t in x:
        for y in t["sol_array"]:
            costs_array.append(y[1])
        
    mean_percentages_simple_1.append(costs_array)

mean_percentages_error_1 = []
for x in mean_percentages_simple_1:
    min = np.min(x)
    mean_err = np.mean([round(100 * np.abs((min - y) / min), 3) for y in x])
    mean_percentages_error_1.append(mean_err)
    

mean_percentages_simple_2 = []
for x in [m2_36, m2_49, m2_64, m2_81]:
    costs_array = []
    for t in x:
        for y in t["sol_array"]:
            costs_array.append(y[1])
        
    mean_percentages_simple_2.append(costs_array)

mean_percentages_error_2 = []
for x in mean_percentages_simple_2:
    min = np.min(x)
    mean_err = np.mean([round(100 * np.abs((min - y) / min), 3) for y in x])
    mean_percentages_error_2.append(mean_err)

print(mean_percentages_error_1)
            
plt.rcParams.update({'font.size': 28})

# Number of nodes
nodes = [36, 49, 64, 81]

# Mean percentage error data for Model 1 and Model 2 (example data)
model1_errors = mean_percentages_error_1  # Mean percentage error for Model 1
model2_errors = mean_percentages_error_2  # Mean percentage error for Model 2

# Set bar width
bar_width = 0.35

# Calculate the position for the bars
bar1_position = np.arange(len(nodes))
bar2_position = [x + bar_width for x in bar1_position]

# Create the grouped bar chart
plt.bar(bar1_position, model1_errors, color='b', width=bar_width, label='Model Advanced')
plt.bar(bar2_position, model2_errors, color='r', width=bar_width, label='Model Simple')

# Set labels and title
plt.xlabel('Number of Nodes')
plt.ylabel('Mean % Error between Samples')
#plt.title('Mean Percentage Error between Best Solution and the others')
plt.xticks(bar1_position + bar_width / 2, nodes)

# Add legend
plt.legend()

# Show plot
plt.show()
