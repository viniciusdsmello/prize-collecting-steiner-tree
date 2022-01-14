"""Module that implements a class to represent an instance of Prize-Collecting Steiner Tree Problem"""
from typing import Set

import networkx as nx


class SteinerTreeProblem():
    def __init__(self, graph=nx.Graph(), terminals=set()):
        self._graph: nx.Graph = graph
        self._terminals: Set[int] = terminals

        self.name: str = None
        self.remark: str = None
        self.creator: str = None
        self.filename: str = None

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, terminals):
        self.terminals = terminals
        self.num_terminals = len(terminals)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self.graph = graph
        self.num_nodes: int = len(self.graph.nodes)
        self.num_edges: int = len(self.graph.edges)
