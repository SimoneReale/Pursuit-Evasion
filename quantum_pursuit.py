from dwave.system import LeapHybridCQMSampler, LeapHybridSampler
from cqm import createCQM, createAdvancedCQM
from new_cqm import createStatesCQM, createMiniModelCQM
from utils import my_token
import pickle
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import hybrid
import sys


def run_hybridCQM_solver(cqm):
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler(token=my_token)
    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label="Quantum Pursuit")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    print(f"Number of correct samples: {len(feasible_sampleset)}")
    
    with open("sampleset_info.pkl", "wb") as f:
        pickle.dump(feasible_sampleset.info, f)
        f.close()
    try:
        sample = feasible_sampleset.first.sample
    except:
        sample = sampleset.first.sample
        original_stdout = sys.stdout
        print("No feasible solution found!")
        with open("violations.txt", "w") as f:
            sys.stdout = f
            print(f"{cqm.violations(sample, skip_satisfied=True)}")
            sys.stdout = original_stdout

    sol = [key for key, val in sample.items() if val > 0]
    return sol


def run_kerberos_solver(cqm):
    response = hybrid.KerberosSampler().sample_qubo(dimod.cqm_to_bqm(cqm))
    response.data_vectors["energy"]


def run_hybridBQM_solver(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm)

    sampler = LeapHybridSampler(token=my_token)       
    sampleset = sampler.sample(bqm, label="Quantum Pursuit BQM")

    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    print(len(feasible_sampleset))
    try:
        sample = feasible_sampleset.first.sample
    except:
        print("\nNo feasible solutions found.")

    print("Feasible solutions found!")
    sol = [key for key, val in sample.items() if val > 0]
    return sol

def run_SimulatedAnnealing_Solver(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm)
    sampleset = dimod.SimulatedAnnealingSampler().sample(bqm)
    sample = sampleset.first.sample
    sol = [key for key, val in sample.items() if val > 0]
    return sol

def run_QPU(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm)
    print(f"\nNum of var: {bqm.num_variables}\nNum of interactions: {bqm.num_interactions}")
    #Advantage_system6.4
    sampler = EmbeddingComposite(
        DWaveSampler(
        token= my_token,
        endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        solver="Advantage2_prototype2.2",
        )
    )
    print("\nStarting QPU: OK")
    sampleset = sampler.sample(bqm)
    sample = sampleset.first.sample
    sol = [key for key, val in sample.items() if val > 0]
    return sol

if __name__ == "__main__":
    cqm, path_prey, costs = createMiniModelCQM()
    sol = run_hybridCQM_solver(cqm)
    # sol = run_hybridBQM_solver(cqm)
    serial = {"sol": sol, "prey": path_prey, "costs": costs}
    with open("items_solution_quantum.pkl", "wb") as f:
        pickle.dump(serial, f)
        f.close()
    print(f"Best Solution: {sol}")
