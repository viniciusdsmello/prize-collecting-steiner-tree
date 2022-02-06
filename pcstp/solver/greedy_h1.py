"""Module that implements a Gready Heuristic for Prize-Collecting Steiner Tree Problem"""
from math import comb
from typing import List, Tuple
import networkx as nx
import networkx.algorithms.components as comp
from pcstp.solver.base import BaseSolver
from itertools import combinations


class GreedyH1(BaseSolver):
    def __init__(self, graph: nx.Graph, terminals: set, **kwargs):
        super().__init__(graph, terminals, **kwargs)
        self._all_terminals_path = []

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree based on Shortest Path Heuristic

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        # For each terminal pair
        terminals_combinations = list(combinations(self.terminals, 2))
        for combination in terminals_combinations:
            terminal_i, terminal_j = combination
            self.log.info('Searching paths between terminals %s and %s', terminal_i, terminal_j)
            paths = self._get_all_paths_between_nodes(
                terminal_i,
                terminal_j)
            min_path, min_cost = self._get_least_cost_path(paths)

            path = {
                "cost": min_cost,
                "path": min_path
            }

            self.log.debug(
                f"Path between {terminal_i} and {terminal_j} ({min_path}) - cost ({min_cost})")

            self._all_terminals_path.append(path)

        # After finding all paths between terminals
        # Sort them based on cost
        self._all_terminals_path.sort(key=lambda path: path['cost'])

        # Add paths to steiner tree
        for terminal_path in self._all_terminals_path:
            nx.add_path(self.steiner_tree, terminal_path['path'])

            # If all terminals are already connected, break the loop
            if self.is_all_terminals_connected():
                break

        # Adds all edges from graph to steiner tree solution
        conn_components = list(comp.connected_components(self.steiner_tree))
        len_conn_components = len(conn_components)
        _history = [len_conn_components]
        while len_conn_components > 1:
            comp1 = list(conn_components[0])
            comp2 = list(conn_components[1])

            if len(_history) > 1 and len(set(_history[-2:])) > 1:
                self.log.info("Found a unstopable loop, trying to add new nodes")
                for node in comp1:
                    neighbors = self._get_neighbors(node)
                    for neighbor in neighbors:
                        if neighbor in comp2:
                            comp1.append(neighbor)
                        else:
                            aux_neighbors = self._get_neighbors(neighbor)
                            for neighbor_of_neighbor in aux_neighbors:
                                comp1.append(neighbor_of_neighbor)
                                
            for j in range(0, len(comp1)):
                for k in range(0, len(comp2)):
                    if self.graph.has_edge(comp1[j], comp2[k]):
                        self.steiner_tree.add_edge(comp1[j], comp2[k])
                        break
            conn_components = list(comp.connected_components(self.steiner_tree))
            len_conn_components = len(conn_components)
            _history.append(len_conn_components)

        # While the steiner tree graph has any cycle, then remove one edge
        while True:
            try:
                cycle = nx.find_cycle(self.steiner_tree)
                self.log.debug(f'Cycle found - {cycle}')
                edge = cycle[0]
                self.log.debug(f'Removing edge {edge}...')
                self.steiner_tree.remove_edge(edge[0], edge[1])
            except:
                break

        self.steiner_cost = self._get_steiner_cost()

        return self.steiner_tree, self.steiner_cost
