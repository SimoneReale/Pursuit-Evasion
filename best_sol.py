import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.lines import Line2D




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



cost_solutions_m1_36 = []
for x in m1_36:
    for sol in x["sol_array"]:
        cost_solutions_m1_36.append(sol[1])

cost_solutions_m1_49 = []
for x in m1_49:
    for sol in x["sol_array"]:
        cost_solutions_m1_49.append(sol[1])

cost_solutions_m1_64 = []
for x in m1_64:
    for sol in x["sol_array"]:
        cost_solutions_m1_64.append(sol[1])

cost_solutions_m1_81 = []
for x in m1_81:
    for sol in x["sol_array"]:
        cost_solutions_m1_81.append(sol[1])


cost_solutions_m2_36 = []
for x in m2_36:
    for sol in x["sol_array"]:
        cost_solutions_m2_36.append(sol[1])

cost_solutions_m2_49 = []
for x in m2_49:
    for sol in x["sol_array"]:
        cost_solutions_m2_49.append(sol[1])

cost_solutions_m2_64 = []
for x in m2_64:
    for sol in x["sol_array"]:
        cost_solutions_m2_64.append(sol[1])

cost_solutions_m2_81 = []
for x in m2_81:
    for sol in x["sol_array"]:
        cost_solutions_m2_81.append(sol[1])



global_best_m1 = {
    36: min(cost_solutions_m1_36),
    49: min(cost_solutions_m1_49),
    64: min(cost_solutions_m1_64),
    81: min(cost_solutions_m1_81)
}

global_best_m2 = {
    36: min(cost_solutions_m2_36),
    49: min(cost_solutions_m2_49),
    64: min(cost_solutions_m2_64),
    81: min(cost_solutions_m2_81)
}


plt.rcParams.update({'font.size': 20}) 

# Step 2: Calculate the percentage difference of the best solution at each iteration from the global best
# Assuming 10 iterations, we need to calculate the best solution per iteration
def calculate_percentage_differences(cost_solutions, global_best):
    # Assuming equal distribution of iterations in the cost_solutions list
    iterations = len(cost_solutions) // 10
    percentage_differences = []
    for i in range(10):  # 10 iterations
        iteration_best = min(cost_solutions[i*iterations:(i+1)*iterations])
        percentage_diff = (iteration_best - global_best) / global_best * 100
        percentage_differences.append(percentage_diff)
    return percentage_differences

# Calculate the percentage differences for M1 and M2 for each node size
percentage_diffs_m1 = {
    nodes: calculate_percentage_differences(globals()[f'cost_solutions_m1_{nodes}'], global_best_m1[nodes])
    for nodes in [36, 49, 64, 81]
}

percentage_diffs_m2 = {
    nodes: calculate_percentage_differences(globals()[f'cost_solutions_m2_{nodes}'], global_best_m2[nodes])
    for nodes in [36, 49, 64, 81]
}




percentage_diffs_m1 = {
    nodes: calculate_percentage_differences(globals()[f'cost_solutions_m1_{nodes}'], global_best_m1[nodes])
    for nodes in [36, 49, 64, 81]
}

percentage_diffs_m2 = {
    nodes: calculate_percentage_differences(globals()[f'cost_solutions_m2_{nodes}'], global_best_m2[nodes])
    for nodes in [36, 49, 64, 81]
}


def plot_model_data(model_name, model_data, global_best, nodes, iterations, bar_width=0.2):
    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes):
        percentage_diffs = model_data[node]
        bars = plt.bar(iterations + i * bar_width, percentage_diffs, width=bar_width, label=f'{node} nodes')
        
        for bar, percentage_diff in zip(bars, percentage_diffs):
            if percentage_diff == 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '\u2B24', ha='center', va='bottom', color=bar.get_facecolor(), fontsize=12)

    plt.title(f'Model "{model_name}"')
    plt.xlabel('Iteration')
    plt.ylabel('Difference from Global Best (%)')
    plt.xticks(iterations + bar_width / 2, iterations)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", markersize=15, linewidth=0, color="lightgrey"))
    labels.append("Optimal solution found")
    plt.legend(labels=labels, handles= handles)
    
    
    plt.tight_layout()
    plt.show()

nodes = [36, 49, 64, 81]
iterations = np.arange(1, 11)

plot_model_data("Advanced", percentage_diffs_m1, global_best_m1, nodes, iterations)
plot_model_data("Simple", percentage_diffs_m2, global_best_m2, nodes, iterations)


