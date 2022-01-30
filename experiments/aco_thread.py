
# # Prize-Collecting Steiner Tree (PCSTP)


# ## Libs Importing


import os
import sys
import time

import networkx as nx
from pcstp.instances.generator import generate_random_steiner
from pcstp.instances.reader import DatReader, SteinlibReader
from pcstp.solver.aco import AntColony
from pcstp.solver.base import computes_steiner_cost
from pcstp.steinertree import SteinerTreeProblem
from pcstp.utils.draw import draw_steiner_graph
from pcstp.utils.graph import preprocessing

sys.path.insert(1, os.path.realpath(os.path.pardir))

# ## Experiments
SEED = 100

G, (nodes, edges, position_matrix, edges_cost, terminals, prizes) = generate_random_steiner(
    num_nodes=25,
    num_edges=20,
    max_node_degree=10,
    min_prize=0,
    max_prize=100,
    num_terminals=5,
    min_edge_cost=0,
    max_edge_cost=10,
    cost_as_length=False,
    max_iter=100,
    seed=SEED
)

stp = SteinerTreeProblem(graph=G, terminals=terminals)

filename = '../data/instances/H/hc6p.stp'

if filename.endswith('.stp'):
    stp_reader = SteinlibReader()
else:
    stp_reader = DatReader()

stp = stp_reader.parser(filename=filename)


print("Nodes: ", len(stp.graph.nodes))
print("Edges: ", len(stp.graph.edges))
print("Terminals: ", stp.terminals)


G, terminals = preprocessing(stp.graph, stp.terminals, verbose=True)


stp_preprocessed = SteinerTreeProblem(graph=G, terminals=terminals)


print("Nodes: ", len(stp_preprocessed.graph.nodes))
print("Edges: ", len(stp_preprocessed.graph.edges))
print("Terminals: ", stp_preprocessed.terminals)

# ## Solution obtained with NetworkX Steiner Tree Approximation Algorithm

start_time = time.time()

nx_steiner_tree = nx.algorithms.approximation.steinertree.steiner_tree(
    stp.graph,
    stp.terminals,
    weight='cost'
)

networkx_duration = time.time() - start_time
networkx_cost = computes_steiner_cost(stp.graph, nx_steiner_tree, stp.terminals)
print(f'Cost: {networkx_cost}')


print(f'Duration: {networkx_duration*1000} ms')


try:
    draw_steiner_graph(
        stp.graph,
        steiner_graph=nx_steiner_tree,
        plot_title=f'NetworkX Implementation - Cost ({networkx_cost}) - Time ({networkx_duration * 1000} ms)',
        node_label='name',
        seed=SEED
    )
except Exception as e:
    print(e)


solver = AntColony(
    graph=stp_preprocessed.graph,
    terminals=stp_preprocessed.terminals,
    iterations=100,
    num_ants=10,
    evaporation_rate=0.2,
    alpha=1.0,
    beta=1.0,
    beta_evaporation_rate=0.2,
    initial_pheromone=0.1,
    pheromone_amount=2,
    pheromone_deposit_strategy='traditional',
    pheromone_initialization_strategy='same_value',
    choose_best=0.3,
    log_level='info',
    early_stopping=None,
    normalize_distance_prize=True,
    allow_edge_perturbation=False,
    seed=SEED
)
steiner_tree, steirner_cost = solver.solve()

print(f'Cost: {steirner_cost}')
print(f'Duration: {solver._duration * 1000} ms')
