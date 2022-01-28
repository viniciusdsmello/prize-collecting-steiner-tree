"""Module that implements a class to represent an instance of Prize-Collecting Steiner Tree Problem"""
import os
import numpy as np
import pandas as pd
from typing import Iterable, Set, Tuple

import networkx as nx


class SteinerTreeProblem():
    def __init__(self, graph=nx.Graph(), terminals=set()):
        self._graph: nx.Graph = graph
        self._terminals: Set[int] = terminals

        self.name: str = None
        self.remark: str = None
        self.creator: str = None
        self.filename: str = None

        self.num_nodes: int = None
        self.num_edges: int = None
        self.num_terminals: int = None

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, terminals: Iterable):
        self._terminals = terminals
        self.num_terminals = len(terminals)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph: nx.Graph):
        self._graph = graph
        self.num_nodes: int = len(self.graph.nodes)
        self.num_edges: int = len(self.graph.edges)

    def to_csv(
        self,
        path: str = 'exports/instances',
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate two dataframes with instance data and write csv files in the export/instances folder.
        One for nodes (nodes.csv) and one for edges (edges.csv)

        Args:
            path (str): Path where output files with be written.
            save (bool): If True, saves file to the path given.
        Returns:
            A tuple containing two dataframes with graph data (nodes_df, edge_df).
        """
        nodes = self._graph.nodes

        nodes_pos = list(dict(self.graph.nodes(data="pos")).values())
        nodes_x = [pos[0] if pos else None for pos in nodes_pos]
        nodes_y = [pos[1] if pos else None for pos in nodes_pos]

        is_terminal = [int(is_terminal) for is_terminal in nx.get_node_attributes(self.graph, 'terminal').values()]
        nodes_prize = [prize for prize in list(nx.get_node_attributes(self.graph, 'prize').values())]

        edges_id = np.arange(0, len(self.graph.edges))
        edges_u = [edge[0] for edge in self.graph.edges]
        edges_v = [edge[1] for edge in self.graph.edges]
        edges_cost = [cost for cost in nx.get_edge_attributes(self.graph, 'cost').values()]

        node_data = {
            "node_id": nodes,
            "x": nodes_x,
            "y": nodes_y,
            "prize": nodes_prize,
            "isTerminal": is_terminal
        }

        edge_data = {
            "edge_id": edges_id,
            "u": edges_u,
            "v": edges_v,
            "cost": edges_cost
        }

        nodes_df = pd.DataFrame(data=node_data)
        edges_df = pd.DataFrame(data=edge_data)

        if save:
            if not os.path.exists(path):
                os.makedirs(path)

            nodes_df.to_csv(f'{path}_nodes.csv', encoding='utf-8', sep=',', index=False)
            edges_df.to_csv(f'{path}_edges.csv', encoding='utf-8', sep=',', index=False)

        return nodes_df, edges_df

    def read_csv(self, filename: str):
        """
         Imports Prize-Collecting Steiner Tree Instance from CSV files
         Args:
             filename(str): Instance filename
         """
        nodes_df = pd.read_csv(f'{filename}_nodes.csv', delimiter=',', encoding='utf-8')
        edges_df = pd.read_csv(f'{filename}_edges.csv', delimiter=',', encoding='utf-8')

        G = nx.Graph()
        terminals = set()
        for i, _ in enumerate(nodes_df.node_id.values):
            x = nodes_df.x[i]
            y = nodes_df.y[i]
            prize = nodes_df.prize[i]

            if nodes_df.isTerminal[i]:
                is_terminal = True
                terminals.add(i)
            else:
                is_terminal = False

            G.add_node(i, pos=(x, y), prize=prize, terminal=is_terminal)

        for j, edge in enumerate(edges_df.edge_id.values):
            u = edges_df.u[j]
            v = edges_df.v[j]
            cost = edges_df.cost[j]

            G.add_edge(u, v, cost=cost)

        self.graph = G
        self.terminals = terminals
