"""Module that implements the Base Solver class for the Prize-Collecting Steiner Tree Problem"""
import time
from typing import List, Set, Tuple

import networkx as nx
import networkx.algorithms.components as comp

import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def computes_steiner_cost(graph: nx.Graph, steiner_tree: nx.Graph, terminals: Set[int]) -> float:
    """Computes the Prize-Collecting Steiner Tree cost

    Args:
        graph (nx.Graph): Instance graph
        steiner_tree (nx.Graph): Solution Graph
        terminals (Set[int]): List of Terminals

    Returns:
        float: Returns the cost of all edges plus all terminals not connected to the tree
    """

    steiner_cost = 0.0

    terminals_not_connected_cost = sum(
        [
            int(graph.nodes[n]['prize']) for n in graph.nodes
            if n in terminals and n not in steiner_tree.nodes
        ]
    )

    edges = nx.get_edge_attributes(graph, 'cost')
    edges_cost = 0
    for edge in steiner_tree.edges:
        if edge in graph.edges:
            edges_cost += edges[tuple(sorted(edge))]

    steiner_cost = edges_cost + terminals_not_connected_cost

    return steiner_cost


class BaseSolver():
    def __init__(self, graph: nx.Graph(), terminals, **kwargs):
        self.graph: nx.Graph = graph
        self.terminals: Set[int] = terminals
        self.steiner_tree = nx.Graph()
        self.steiner_cost: float = None

        self._start_time = None
        self._end_time = None
        self._duration = None

        logging.basicConfig(
            level=str(kwargs.get("log_level", 'info')).upper(),
            format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger('solver')

    def _get_all_paths_between_nodes(self, u: int, v: int) -> list:
        """
        Given a pair of nodes, finds all simple paths between them.

        Args:
            u (int): First node index
            v (int): Second node index

        A *simple path* in a graph is a nonempty sequence of nodes in which
        no node appears more than once in the sequence, and each adjacent
        pair of nodes in the sequence is adjacent in the graph.

        This algorithm uses a modified depth-first search to generate the
        paths [1]_.  A single path can be found in $O(V+E)$ time but the
        number of simple paths in a graph can be very large, e.g. $O(n!)$ in
        the complete graph of order $n$.

        References
        ----------
        .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
        Addison Wesley Professional, 3rd ed., 2001.
        """
        self.log.debug(f"Finding all paths between {u} and {v}...")
        
        # TODO: Try different algoritms in order to find paths between nodes.
        all_paths = list(nx.all_simple_paths(G=self.graph, source=u, target=v))

        return all_paths

    def _get_path_cost(self, path: List[int]) -> float:
        """Given a path computes its cost

        Args:
            path (List[int]): Path as a sequential list of nodes 

        Returns:
            float: Returns the total cost of a given path
        """
        path_cost = 0.0

        # Path cost is composed by all edge distances plus all the terminals prizes not present

        terminals_not_connected_cost = sum(
            [
                int(self.graph.nodes[n]['prize'])
                for n in self.graph.nodes
                if n in self.terminals and n not in path
            ]
        )
        edges_cost = nx.path_weight(self.graph, path, weight='cost')

        path_cost = edges_cost + terminals_not_connected_cost

        return path_cost

    def _get_steiner_cost(self) -> float:
        """Returns the cost for the steiner tree solution

        Returns:
            float: Returns the total cost of a given path
        """
        steiner_cost = computes_steiner_cost(
            self.graph,
            self.steiner_tree,
            self.terminals
        )

        return steiner_cost

    def _get_least_cost_path(self, paths: List[List[int]]) -> Tuple[List[int], float]:
        """Given a list of Paths, finds the minimium path and its cost

        Args:
            paths (List[List[int]]): [description]

        Returns:
            Tuple[List[int], float]: Returns the minimium path and its cost
        """
        min_cost_path: List[int] = []
        min_cost = float("inf")

        for path in paths:
            cost = self._get_path_cost(path)
            if cost < min_cost:
                min_cost = cost
                min_cost_path = path
        return min_cost_path, min_cost

    def is_all_terminals_connected(self) -> bool:
        """
        Method that check if all steiner terminals are connected in the steiner tree

        Returns:
            bool: Returns True if all terminals are connected and False if there terminals not connecteds
        """
        for i in range(len(self.terminals)):
            for j in range(i+1, len(self.terminals)):
                try:
                    # Check if there are simple paths to all pair of terminals
                    paths = list(nx.all_simple_paths(self.steiner_tree, self.terminals[i], self.terminals[j]))
                    if len(paths) == 0:
                        return False
                except:
                    return False
        return True

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        raise NotImplementedError

    def solve(self) -> Tuple[nx.Graph, int]:
        """Solves the Prize-Collecting Steiner Tree

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        self._start_time = time.time()
        steiner_tree, steiner_cost = self._solve()
        self._end_time = time.time()
        self._duration = self._end_time - self._start_time
        self.log.debug(f"Runtime of the program is {self._duration * 1000} miliseconds")

        return steiner_tree, steiner_cost
