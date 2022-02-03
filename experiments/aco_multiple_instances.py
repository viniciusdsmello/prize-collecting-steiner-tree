# %% [markdown]
# # Prize-Collecting Steiner Tree (PCSTP)

# %% [markdown]
# ## Libs Importing

# %%
import sys
import os
import time
import networkx as nx

sys.path.insert(1, os.path.realpath(os.path.pardir))

# %%
import multiprocessing

NUM_PROCESSES = multiprocessing.cpu_count()
print("Number of cpu : ", NUM_PROCESSES)

# %%
from pcstp.instances.generator import generate_random_steiner
from pcstp.instances.reader import SteinlibReader, DatReader

from pcstp.steinertree import SteinerTreeProblem
from pcstp.solver.base import computes_steiner_cost
from pcstp.solver.aco import AntColony
from pcstp.solver.greedy_h1 import GreedyH1

from pcstp.utils.graph import preprocessing
from pcstp.utils.draw import draw_steiner_graph

# %% [markdown]
# ## Experiments

# %%
SEED = 100

# %% [markdown]
# ## Solution obtained with Ant Colony Optimization

# %%
import glob

INSTANCES_PATH_PREFIX = './data/instances/benchmark/SteinCD'
NUM_EXPERIMENTS_PER_INSTANCE = 5

all_files = glob.glob(os.path.join(INSTANCES_PATH_PREFIX, '*'))

files = all_files

aco_history = []
solutions = {}
for filename in files:
    if filename.endswith('.xlsx') or filename.endswith('.csv'): continue
    if filename.endswith('.stp'):
        stp_reader = SteinlibReader()
    else:
        stp_reader = DatReader()

    print(f"Reading: {filename}")
    stp = stp_reader.parser(filename=filename)
    G, terminals = preprocessing(stp.graph, stp.terminals)
    stp_preprocessed = SteinerTreeProblem(graph=G, terminals=terminals)

    def run_experiment(experiment: int):
        aco_params = dict(
            iterations=50,
            num_ants=len(terminals),
            evaporation_rate=0.5,
            alpha=1.0,
            beta=3.0,
            # beta_evaporation_rate=0.2,
            initial_pheromone=0.5,
            pheromone_amount=2.0,
            pheromone_deposit_strategy='traditional',
            pheromone_initialization_strategy='same_value',
            choose_best=0.2,
            log_level='info',
            early_stopping=10,
            normalize_distance_prize=False,
            allow_edge_perturbation=False,
            ant_max_moves=len(stp_preprocessed.graph.nodes),
            seed=SEED * experiment
        )
        solver = AntColony(
            graph=stp_preprocessed.graph,
            terminals=stp_preprocessed.terminals,
            **aco_params
        )
        steiner_tree, steiner_cost = solver.solve()

        history = {
            "filename": filename,
            "experiment": experiment,
            "num_nodes": stp.num_nodes,
            "num_edges": stp.num_edges,
            "num_nodes_after_preprocessing": len(stp_preprocessed.graph.nodes),
            "num_edges_after_preprocessing": len(stp_preprocessed.graph.edges),
            "terminals": stp.num_terminals,
            "steiner_cost": steiner_cost,
            "duration": solver._duration
        }
        history.update(aco_params)
        return history, solver

    experiments = range(1, NUM_EXPERIMENTS_PER_INSTANCE+1)

    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        experiments_results = p.map(run_experiment, experiments)
    
    aco_history.extend([result[0] for result in experiments_results])
    solutions[filename] = [result[1] for result in experiments_results]

import pandas as pd

df_score_aco = pd.DataFrame.from_dict(aco_history)
df_score_aco.to_csv(os.path.join(INSTANCES_PATH_PREFIX, 'ACO.csv'))

# %%
df_score_aco


