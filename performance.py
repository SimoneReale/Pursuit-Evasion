import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats
from utils import n_of_nodes, n_time, n_preys





def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if __name__ == "__main__":
  with open(f"MSimple_mver_{1}_n{n_of_nodes}_p{n_preys}_t{n_time}.pkl", "rb") as fp: 
        pack = pickle.load(fp)
        print(pack)