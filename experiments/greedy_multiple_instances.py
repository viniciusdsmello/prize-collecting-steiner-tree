# %% [markdown]
# # Prize-Collecting Steiner Tree (PCSTP)

# %% [markdown]
# ## Libs Importing

import glob
import multiprocessing
import os
import random
import sys
import numpy as np
# %%
import pandas as pd
from pcstp.instances.reader import DatReader, SteinlibReader
from pcstp.solver.greedy_h1 import GreedyH1
from pcstp.steinertree import SteinerTreeProblem
from pcstp.utils.graph import preprocessing

sys.path.insert(1, os.path.realpath(os.path.pardir))

# %%

NUM_PROCESSES = multiprocessing.cpu_count()
print("Number of cpu : ", NUM_PROCESSES)

# %%


# %% [markdown]
# ## Experiments

# %%
SEED = 100

# %% [markdown]
#
# # Greedy

# %%

INSTANCES_PATH_PREFIX = '../data/instances/benchmark/PCSPG-CRR'
NUM_EXPERIMENTS_PER_INSTANCE = 5

all_files = glob.glob(os.path.join(INSTANCES_PATH_PREFIX, '*'))

files = all_files

greedy_history = []
solutions = {}

for filename in files:
    if filename.endswith('.xlsx') or filename.endswith('.csv'):
        continue
    if filename.endswith('.stp'):
        stp_reader = SteinlibReader()
    else:
        stp_reader = DatReader()

    print(f"Reading: {filename}")
    stp = stp_reader.parser(filename=filename)
    G, terminals = preprocessing(stp.graph, stp.terminals)
    stp_preprocessed = SteinerTreeProblem(graph=G, terminals=terminals)
    print("Nodes: ", len(stp_preprocessed.graph.nodes))
    print("Edges: ", len(stp_preprocessed.graph.edges))
    # print("Terminals: ", stp_preprocessed.terminals)

    def run_experiment(experiment: int):
        if SEED:
            np.random.seed(SEED*experiment)
            random.seed(SEED*experiment)
        solver = GreedyH1(stp_preprocessed.graph, list(stp_preprocessed.terminals), log_level='info')
        steiner_tree, greedy_cost = solver.solve()
        print(f'Cost: {greedy_cost} ')

        history = {
            "filename": filename,
            "experiment": experiment,
            "num_nodes": stp.num_nodes,
            "num_edges": stp.num_edges,
            "num_nodes_after_preprocessing": len(stp_preprocessed.graph.nodes),
            "num_edges_after_preprocessing": len(stp_preprocessed.graph.edges),
            "terminals": stp.num_terminals,
            "steiner_cost": greedy_cost,
            "duration": solver._duration
        }
        return history, solver

    experiments = range(1, NUM_EXPERIMENTS_PER_INSTANCE+1)

    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        experiments_results = p.map(run_experiment, experiments)

    greedy_history.extend([result[0] for result in experiments_results])
    solutions[filename] = [result[1] for result in experiments_results]


# %%

df_score_greedy = pd.DataFrame.from_dict(greedy_history)
df_score_greedy.to_csv(os.path.join(INSTANCES_PATH_PREFIX, 'GREEDY.csv'))
