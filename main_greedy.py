import os
import json
import numpy as np
import random
import argparse
import shutil

from collections import Counter

import matplotlib.pyplot as plt
from utils.trace_loader import TraceLoader
from Data_Set.Generation.algorithms.RWA_SPF_FF_CUSTOM import RWA_SPF_FF

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="MLP_FC")
ap.add_argument("-k", "--kpath", default=3)
ap.add_argument("-n", "--nodes", default=10)
ap.add_argument("-d", "--degree", default=5)
ap.add_argument("-w", "--wavelength", default=4)
ap.add_argument("-c", "--clearlogs", default=True)
ap.add_argument("-r", "--rewardfunction", default=1)
ap.add_argument("-s", "--stats", default=True)
args = vars(ap.parse_args())


def main():

    nodes = [args["nodes"]]
    degree = [args["degree"]]
    netw_i = 0
    lpMat_i = 0
    maxUtilPerNodePair = [0.3]
    nbNetwork = 10
    traffic="uniform"

    k_path_generated = args["kpath"]
    gen_state = 1
    numWavelengths = args["wavelength"]

    topology_path = f"Data_set/raw_data/{nodes[0]}node/{degree[0]}degree/{netw_i}_instance/"
    traces_path = f"Data_set/raw_data/{nodes[0]}node/{lpMat_i}_traff_matrix/{int(maxUtilPerNodePair[0]*100)}_ut/"

    topology = json.load(open(topology_path+"topology.json"))
    traces = TraceLoader(traces_path)

    actions = []
    results = []
    for i, trace in enumerate(traces):
        a = RWA_SPF_FF(topology, trace, numWavelengths, k_path_generated)
        c = Counter(a)
        actions.append([c[t] for t in range(k_path_generated+1)])
        results.append([sum(c[t+1] for t in range(k_path_generated)),c[0],0])
        print(f"Trace {i+1}:\n"+"\n".join([f"\t{a}: {b}" for a,b in enumerate(actions[-1])]))

    filename = f"n{nodes[0]}_d{degree[0]}_RWA.json"
    h = {"results": results, "actions": actions }
    with open(f"results/{filename}","w") as file:
        file.write(json.dumps(h))
    print("Results saved")





main()
