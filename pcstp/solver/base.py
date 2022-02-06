"""Module that implements the Base Solver class for the Prize-Collecting Steiner Tree Problem"""
import time
from typing import Iterable, List, Set, Tuple

import networkx as nx
import networkx.algorithms.components as comp

import logging


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
            float(graph.nodes[n]['prize']) for n in graph.nodes
            if n in terminals and n not in steiner_tree.nodes
        ]
    )

    edges_cost = 0.0
    for edge in steiner_tree.edges:
        if edge in graph.edges:
            edges_cost += graph.edges[edge]['cost']

    steiner_cost = edges_cost + terminals_not_connected_cost

    return steiner_cost


class BaseSolver():
    def __init__(self, graph: nx.Graph(), terminals: Iterable[int], **kwargs):
        self.graph: nx.Graph = graph
        self.terminals: List[int] = list(terminals)
        self.steiner_tree = nx.Graph()
        self.steiner_cost: float = None

        self._start_time = None
        self._end_time = None
        self._duration = None

        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        logging.basicConfig(
            level=str(kwargs.get("log_level", 'info')).upper(),
            format='%(asctime)s - [%(filename)s:%(lineno)d] - %(threadName)s - %(levelname)s - %(message)s',
            force=True
        )
        self.log = logging.getLogger('solver')

    def _get_all_paths_between_nodes(self, u: int, v: int) -> list:
        """
        Given a pair of nodes, finds all simple paths between them.

        Args:
            u (int): First node index
            v (int): Second node index
        """
        self.log.debug(f"Finding all paths between {u} and {v}...")

        # TODO: Try different algoritms in order to find paths between nodes.
        all_paths = list(nx.all_shortest_paths(G=self.graph, source=u, target=v, weight='cost'))

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

    def _get_steiner_cost(self, steiner_tree: nx.Graph = None) -> float:
        """Returns the cost for the steiner tree solution

        Args:
            steiner_tree (nx.Graph, optional): Steiner Tree solution to be evaluated. If not passed, it evaluates the attribute
            steiner_tree. Defaults to None.

        Returns:
            float: Returns the total cost of a given path
        """
        if steiner_tree is None:
            steiner_tree = self.steiner_tree

        steiner_cost = computes_steiner_cost(
            self.graph,
            steiner_tree,
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
        self.log.debug("Checking if all terminals are connected...")

        terminals_connected = []
        for terminal in self.terminals:
            terminals_connected.append(terminal in self.steiner_tree.nodes)
        return all(terminals_connected)

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

    def _process_cycles(self, steiner_tree: nx.Graph = None) -> nx.Graph:
        """
        Process steiner_tree in order to find and remove any cycle found

        Args:
            steiner_tree (nx.Graph, optional): Steiner Tree solution. If not set, the attribute steiner_tree will be used.
                Defaults to None.
        Returns:
            (nx.Graph): Returns the steiner_tree solution without cycles.
        """
        self.log.debug("Checking cycles...")
        all_edges_cost = nx.get_edge_attributes(self.graph, 'cost')
        
        if steiner_tree is None:
            steiner_tree = self.steiner_tree
        
        check_cycles = True
        while check_cycles:
            try:
                cycle = nx.find_cycle(steiner_tree)
                self.log.debug(f'Cycle found - {cycle}')

                cycle_edges_cost = list(filter(lambda edge: edge in cycle, all_edges_cost))
                # TODO: Sort edges by cost
                # cycle_edges_cost.sort(key=lambda edge: edge['cost'])

                edge = cycle_edges_cost[-1]
                self.log.debug(f'Removing edge {edge}...')
                steiner_tree.remove_edge(*edge)
            except nx.NetworkXNoCycle:
                self.log.debug(f'No cycle found')
                check_cycles = False
            except Exception as e:
                self.log.exception("Error %s", e)
        return steiner_tree