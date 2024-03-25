from dwave.system import LeapHybridCQMSampler, LeapHybridSampler
from cqm import createCQM, createAdvancedCQM
from new_cqm import createStatesCQM, createMiniModelCQM
from utils import my_token, n_of_nodes, n_time, n_preys
import pickle
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.samplers import SteepestDescentSolver, TabuSampler
from dwave.system.composites import EmbeddingComposite
from dwave.preprocessing.presolve import Presolver
import hybrid
import sys
import dwave.inspector
from alive_progress import alive_bar


# with open("sampleset_info.pkl", "wb") as f:
#         pickle.dump(feasible_sampleset.info, f)
#         f.close()
#     try:
#         sample = presolve.restore_samples(sampleset.first.sample)
#     except:
#         sample = presolve.restore_samples(sampleset.first.sample)
#         original_stdout = sys.stdout
#         print("No feasible solution found!")
#         with open("violations.txt", "w") as f:
#             sys.stdout = f
#             print(f"{cqm.violations(sample, skip_satisfied=True)}")
#             sys.stdout = original_stdout


def run_hybrid_Model_Advanced(cqm, costs):
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler(token=my_token)
    presolve = Presolver(cqm)
    print("\nPresolving...")
    presolve.apply()
    reduced_cqm = presolve.detach_model()

    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(reduced_cqm, label="Quantum Pursuit hybrid cqm")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    print("Qpu access time: ", sampleset.info["qpu_access_time"])
    print("Total run time: ", sampleset.info["run_time"])

    print(
        f"Total number of samples: {len(sampleset)}\nNumber of correct samples: {len(feasible_sampleset)}\nPercentage: {100 * (len(feasible_sampleset) / len(sampleset))}%"
    )

    solutions_array = []

    for x in feasible_sampleset:
        restored = presolve.restore_samples(x)
        sol = []
        for val, key in zip(restored[0][0], restored[1]):
            if val > 0 and key[0] == 'x':
                _ , i , j ,t = key.split("_")
                sol.append((int(i), int(j), int(t)))
        solutions_array.append((sol, calculateCostAdvanced(sol, costs)))

    pack = {
        "qpu_access_time": sampleset.info["qpu_access_time"],
        "run_time": sampleset.info["run_time"],
        "total_n_of_samples": len(sampleset),
        "n_of_feasible": len(feasible_sampleset),
        "sol_array": solutions_array
    }

    return pack



def run_hybrid_Model_Simple(cqm, costs):
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler(token=my_token)

    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label="Quantum Pursuit hybrid cqm")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    print("Qpu access time: ", sampleset.info["qpu_access_time"])
    print("Total run time: ", sampleset.info["run_time"])

    print(
        f"Total number of samples: {len(sampleset)}\nNumber of correct samples: {len(feasible_sampleset)}\nPercentage: {100 * (len(feasible_sampleset) / len(sampleset))}%"
    )

    solutions_array = []

    for x in feasible_sampleset:
        sol = []
        for key, val in x.items():
            if val > 0:
                sol.append(key)
        solutions_array.append((sol, calculateCostMiniModel(sol, costs)))

    pack = {
        "qpu_access_time": sampleset.info["qpu_access_time"],
        "run_time": sampleset.info["run_time"],
        "total_n_of_samples": len(sampleset),
        "n_of_feasible": len(feasible_sampleset),
        "sol_array": solutions_array
    }

    return pack



# def run_hybridCQM_solver(cqm):
#     # Initialize the CQM solver
#     sampler = LeapHybridCQMSampler(token=my_token)
#     print("\nStarting hybrid sampler: OK")
#     # Solve the problem using the CQM solver
#     sampleset = sampler.sample_cqm(cqm, label="Quantum Pursuit")
#     feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
#     print(feasible_sampleset.record)
#     print(f"Number of correct samples: {len(feasible_sampleset)}")

#     with open("sampleset_info.pkl", "wb") as f:
#         pickle.dump(feasible_sampleset.info, f)
#         f.close()
#     try:
#         sample = feasible_sampleset.first.sample
#     except:
#         sample = sampleset.first.sample
#         original_stdout = sys.stdout
#         with open("violations.txt", "w") as f:
#             sys.stdout = f
#             print(f"{cqm.violations(sample, skip_satisfied=True)}")
#             sys.stdout = original_stdout

#     sol = [key for key, val in sample.items() if val > 0]
#     return sol


def calculateCostAdvanced(solution, costs):
    total_cost = 0
    for s in solution:
        total_cost += costs[s]
    return total_cost


def calculateCostMiniModel(data, costs):
    # Define a custom key function to extract the time 't' from the string
    def extract_t(item):
        _, _, t = item.split("_")
        return int(t)

    sorted_data = sorted(data, key=extract_t)
    total_cost = 0
    for item in range(len(data) - 1):
        _, i, _ = sorted_data[item].split("_")
        _, j, t = sorted_data[item + 1].split("_")
        total_cost += costs[(int(i), int(j), int(t))]

    return total_cost


if __name__ == "__main__":
    
    cqm, path_prey, costs = createAdvancedCQM()

    # solution_1 = [tuple(map(int, s.split('_')[1:])) for s in run_hybridCQM_solver(cqm) if s.startswith('x')]
    # solution_2 = [tuple(map(int, s.split('_')[1:])) for s in run_hybridCQM_solver(cqm) if s.startswith('x')]
    # cost1 = calculateCostAdvanced(solution_1, costs)
    # cost2 = calculateCostAdvanced(solution_2, costs)

    #pack_1 = run_hybridCQM_solver(cqm, costs)

    lst_pack = []
    n_of_iteration = 10
    with alive_bar(n_of_iteration) as bar:
        for _ in range(n_of_iteration):
            lst_pack.append(run_hybrid_Model_Advanced(cqm, costs))
            bar()
    
    with open(f"mver_{n_of_iteration}_n{n_of_nodes}_p{n_preys}_t{n_time}.pkl", "wb") as f:
        pickle.dump(lst_pack, f)
        f.close()

    # with open(f"mver_{n_of_iteration}_n{n_of_nodes}_p{n_preys}_t{n_time}.pkl", "rb") as fp: 
    #     pack = pickle.load(fp)



    # solution_2 = run_hybridCQM_solver(cqm)

    # cost1 = calculateCostMiniModel(solution_1, costs)
    # cost2 = calculateCostMiniModel(solution_2, costs)

    # print(f"Cost1: {cost1}\nCost2: {cost2}")

    # print(f"First sol: {solution_1}\nSecond: {solution_2}")
