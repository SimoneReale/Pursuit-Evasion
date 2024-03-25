from dwave.system import LeapHybridCQMSampler, LeapHybridSampler
from cqm import createCQM, createAdvancedCQM
from new_cqm import createStatesCQM, createMiniModelCQM
from utils import my_token
import pickle
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.samplers import SteepestDescentSolver, TabuSampler
from dwave.system.composites import EmbeddingComposite
from dwave.preprocessing.presolve import Presolver
import hybrid
import sys
import dwave.inspector


# def run_hybridCQM_solver(cqm):
#     # Initialize the CQM solver
#     sampler = LeapHybridCQMSampler(token=my_token)
#     presolve = Presolver(cqm)
#     print("Presolving")
#     presolve.apply()
#     reduced_cqm = presolve.detach_model()
    
#     print("\nStarting hybrid sampler: OK")
#     # Solve the problem using the CQM solver
#     sampleset = sampler.sample_cqm(reduced_cqm, label="Quantum Pursuit")

#     feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

#     print(f"Number of correct samples: {len(feasible_sampleset)}\nTotal number of samples: {len(sampleset)}")
    
#     # with open("sampleset_info.pkl", "wb") as f:
#     #     pickle.dump(feasible_sampleset.info, f)
#     #     f.close()

#     sample = presolve.restore_samples(sampleset.first.sample)

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

#     sol = [key for key, val in sample.items() if val > 0]

#     return sol


def run_hybridCQM_solver(cqm):
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler(token=my_token)
    presolve = Presolver(cqm)
    print("Presolving")
    presolve.apply()
    reduced_cqm = presolve.detach_model()
    print("\nStarting hybrid sampler: OK")
    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(reduced_cqm, label="Quantum Pursuit")
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    print(f"Number of correct samples: {len(feasible_sampleset)}")
    
    with open("sampleset_info.pkl", "wb") as f:
        pickle.dump(feasible_sampleset.info, f)
        f.close()
    try:
        sample = presolve.restore_samples(sampleset.first.sample)
    except:
        sample = presolve.restore_samples(sampleset.first.sample)
        original_stdout = sys.stdout
        print("No feasible solution found!")
        with open("violations.txt", "w") as f:
            sys.stdout = f
            print(f"{cqm.violations(sample, skip_satisfied=True)}")
            sys.stdout = original_stdout

    sol = [key for val, key in zip(sample[0][0], sample[1]) if val > 0 and key[0] == 'x']

    return sol


def run_kerberos_solver(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1000)
    sampler = hybrid.KerberosSampler()
    sampleset = sampler.sample(bqm,num_reads = 5, max_iter=200, max_time=None, convergence=8, energy_threshold=None,
             sa_reads=10, sa_sweeps=30000, tabu_timeout=2500,
             qpu_reads=500, qpu_sampler=None, qpu_params=None,
             max_subproblem_size=50)
    sample = sampleset.first.sample

    feasible = []
    for x in sampleset:
        sample_cqm = invert(x)
        if cqm.check_feasible(sample_cqm):
            feasible.append(sample_cqm)

    if not feasible:
        sample_cqm = invert(sampleset.first.sample)
        print("\nNessuna feasible")
        print(f"\n{cqm.violations(sample_cqm, skip_satisfied=True)}\n")
    else:
        print(f"Ho {len(feasible)} soluzioni!")
        print(f"This is the best: {feasible[0]}")


    sol = [key for key, val in sample.items() if val > 0]
    return sol
    


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
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier = 3000)
    sampleset = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=20, num_sweeps=5000)
    sample = sampleset.first.sample
    feasible = []
    for x in sampleset:
        sample_cqm = invert(x)
        if cqm.check_feasible(sample_cqm):
            feasible.append(sample_cqm)

    if not feasible:
        sample_cqm = invert(sampleset.first.sample)
        print("\nNessuna feasible")
        print(f"\n{cqm.violations(sample_cqm, skip_satisfied=True)}\n")
    else:
        print(f"Ho {len(feasible)} soluzioni!")
        print(f"This is the best: {feasible[0]}")
    sample_cqm = invert(sample)
    print(f"{cqm.violations(sample_cqm, skip_satisfied=True)}")
    sol = [key for key, val in sample.items() if val > 0]
    return sol

def run_Tabu(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=3000)
    feasible = []
    iter = 1
    while not feasible:
        print("Iteration ", iter)
        iter += 1
        sampleset = TabuSampler().sample(bqm, timeout=30000, num_reads=1)
        sample = sampleset.first.sample
        
        for x in sampleset:
            sample_cqm = invert(x)
            if cqm.check_feasible(sample_cqm):
                feasible.append(sample_cqm)

        if not feasible:
            sample_cqm = invert(sampleset.first.sample)
            print("\nNessuna feasible")
            print(f"\n{cqm.violations(sample_cqm, skip_satisfied=True)}\n")
        else:
            print(f"Ho {len(feasible)} soluzioni!")
            print(f"This is the best: {feasible[0]}")
        sample_cqm = invert(sample)
        print(f"{cqm.violations(sample_cqm, skip_satisfied=True)}")
        sol = [key for key, val in sample.items() if val > 0]
    return sol

def run_Steepest(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm)
    print("\nStaring Steepest descent")
    sampleset = SteepestDescentSolver().sample(bqm)
    sample = sampleset.first.sample
    sample_cqm = invert(sample)
    print(f"{cqm.violations(sample_cqm, skip_satisfied=True)}")
    sol = [key for key, val in sample.items() if val > 0]
    return sol


def run_QPU(cqm):
    presolve = Presolver(cqm)
    presolve.apply()
    reduced_cqm = presolve.detach_model()

    bqm, invert = dimod.cqm_to_bqm(reduced_cqm, lagrange_multiplier=600)

    print(f"\nNum of var: {bqm.num_variables}\nNum of interactions: {bqm.num_interactions}")
    #Advantage_system6.4
    #"Advantage2_prototype2.2"
    sampler = EmbeddingComposite(
        DWaveSampler(
        token= my_token,
        endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        solver="Advantage2_prototype2.2",
        )
    )
    print("\nStarting QPU: OK")

    sampleset = sampler.sample(bqm, chain_strength=500)
    dwave.inspector.show(sampleset)
    new_reduced_sample = sampleset.first.sample
    sample = invert(new_reduced_sample)
    
    feasible = []
    for x in sampleset:
        sample_cqm = invert(x)
        if reduced_cqm.check_feasible(sample_cqm):
            feasible.append(sample_cqm)

    if not feasible:
        sample_cqm = invert(sampleset.first.sample)
        print("\nNessuna feasible")
        sample_first_cqm = presolve.restore_samples(sample_cqm)
        print(f"\nViolazioni nuova cqm: {reduced_cqm.violations(sample_cqm, skip_satisfied=True)}\n")
        print(f"\nViolazioni vecchia cqm: {cqm.violations(sample_first_cqm, skip_satisfied=True)}\n")
    else:
        print(f"Ho {len(feasible)} soluzioni!")
        print(f"This is the best: {feasible[0]}")

    sample_cqm = invert(sample)
    sol = [key for key, val in sample.items() if val > 0]

    return sol




def prova(cqm):
    presolve = Presolver(cqm)
    presolve.apply()
    reduced_cqm = presolve.detach_model()

    bqm, invert = dimod.cqm_to_bqm(reduced_cqm, lagrange_multiplier=3000)
    bqm_non_opt, invert_non_opt = dimod.cqm_to_bqm(cqm, lagrange_multiplier=3000)
    print(f"\nOPTIMAL    Num of var: {bqm.num_variables}\nNum of interactions: {bqm.num_interactions}")
    print(f"\nSUBOPTIMAL Num of var: {bqm_non_opt.num_variables}\nNum of interactions: {bqm_non_opt.num_interactions}")


if __name__ == "__main__":

    art = """   ____                            _                            _____                          _     
  / __ \                          | |                          / ____|                        | |    
 | |  | |  _   _    __ _   _ __   | |_   _   _   _ __ ___     | |  __   _ __    __ _   _ __   | |__  
 | |  | | | | | |  / _` | | '_ \  | __| | | | | | '_ ` _ \    | | |_ | | '__|  / _` | | '_ \  | '_ \ 
 | |__| | | |_| | | (_| | | | | | | |_  | |_| | | | | | | |   | |__| | | |    | (_| | | |_) | | | | |
  \___\_\  \__,_|  \__,_| |_| |_|  \__|  \__,_| |_| |_| |_|    \_____| |_|     \__,_| | .__/  |_| |_|
                                                                                      | |            
                                                                                      |_|            """
    

    art2 ="""\n  _____                                 _   _   
 |  __ \                               (_) | |  
 | |__) |  _   _   _ __   ___   _   _   _  | |_ 
 |  ___/  | | | | | '__| / __| | | | | | | | __|
 | |      | |_| | | |    \__ \ | |_| | | | | |_ 
 |_|       \__,_| |_|    |___/  \__,_| |_|  \__|
                                                
                                                """
    

    print(art, art2)


    cqm, path_prey, costs = createAdvancedCQM()
    #sol = run_QPU(cqm)
    sol = run_hybridCQM_solver(cqm)
    # print("\n\nSecond iteration")
    # sol2 = run_hybridCQM_solver(cqm)

    serial = {"sol": sol, "prey": path_prey, "costs": costs}
    with open("items_solution_quantum.pkl", "wb") as f:
        pickle.dump(serial, f)
        f.close()
    print(f"First sol: {sol}")
    
    
