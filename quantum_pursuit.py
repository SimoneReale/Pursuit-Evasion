from dwave.system import LeapHybridCQMSampler
from cqm import createCQM
from utils import my_token
import pickle


def run_hybrid_solver(cqm):
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
    return sol


if __name__ == "__main__":
    cqm, path_prey, costs = createCQM()
    sol = run_hybrid_solver(cqm)
    serial = {"sol": sol, "prey": path_prey, "costs": costs}
    with open("items_solution_quantum.pkl", "wb") as f:  # open a text file
        pickle.dump(serial, f)  # serialize the list
        f.close()

    print(f"Best Solution: {sol}")
    print(f"Prey path: {path_prey}")
