import matplotlib.pyplot as plt
import numpy as np
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


# total_times_all = []
# for x in [m1_36, m1_49, m1_64, m1_81]:
#     run_times = [y["qpu_access_time"] * 10**(-6)  for y in x]
#     total_times_all.append(run_times)

total_times_all = []
for x in [m2_36, m2_49, m2_64, m2_81]:
    run_times = [y["qpu_access_time"] * 10**(-6) for y in x]
    total_times_all.append(run_times)
    
        

        
print(total_times_all)


plt.rcParams.update({'font.size': 28})
total_runtimes = {
    36: total_times_all[0],  # Example data for 36 nodes
    49: total_times_all[1],  # Example data for 59 nodes
    64: total_times_all[2],  # Example data for 64 nodes
    81: total_times_all[3],  # Example data for 81 nodes
}


data = [total_runtimes[node] for node in total_runtimes]


plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=total_runtimes.keys(), patch_artist=True)


#plt.title('Box Plot of Total Run Time for Different Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('Total QPU Access Time (seconds)')
plt.grid(True)
plt.show()
