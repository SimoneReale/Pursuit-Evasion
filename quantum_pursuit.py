from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
from cqm import createCQM, createAdvancedCQM
from utils import my_token
import pickle
import dimod
import hybrid


def run_hybridCQM_solver(cqm):
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler(token=my_token)
    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label="Quantum Pursuit")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    try:
        sample = feasible_sampleset.first.sample
    except:
        print("\nNo feasible solutions found.")
        exit()

    print("Feasible solutions found!")
    sol = [key for key, val in sample.items() if val > 0]
    print("\n\nOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(sol)
    return sol


def run_kerberos_solver(cqm):
    response = hybrid.KerberosSampler().sample_qubo(dimod.cqm_to_bqm(cqm))   
    response.data_vectors['energy']
    

def run_hybridBQM_solver(cqm):
    sampler_BQM = LeapHybridBQMSampler(token=my_token)
    bqm, invert = dimod.cqm_to_bqm(cqm)
    qubo, invert = bqm.to_qubo()

    
    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler_BQM.sample_qubo(qubo, label="Quantum Pursuit")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    try:
        sample = feasible_sampleset.first.sample
    except:
        print("\nNo feasible solutions found.")
        exit()

    print("Feasible solutions found!")
    sol = [key for key, val in sample.items() if val > 0]
    print("\n\nOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(sol)
    return sol


if __name__ == "__main__":
    cqm, path_prey, costs = createAdvancedCQM()
    sol = run_hybridCQM_solver(cqm)
    # sol = run_hybridBQM_solver(cqm)
    serial = {"sol": sol, "prey": path_prey, "costs": costs}
    with open("items_solution_quantum.pkl", "wb") as f:  # open a text file
        pickle.dump(serial, f)  # serialize the list
        f.close()
   
    print(f"Best Solution: {sol}")
    print(f"Prey path: {path_prey}")
