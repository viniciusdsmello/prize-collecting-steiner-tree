"""Module that implements a Gready Heuristic for Prize-Collecting Steiner Tree Problem"""
from typing import List, Tuple
import networkx as nx
import networkx.algorithms.components as comp
from pcstp.solver.base import BaseSolver


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
        for terminal_i in range(len(self.terminals)):
            for terminal_j in range(terminal_i+1, len(self.terminals)):
                paths = self._get_all_paths_between_nodes(
                    list(self.terminals)[terminal_i],
                    list(self.terminals)[terminal_j])
                min_path, min_cost = self._get_least_cost_path(paths)

                path = {
                    "cost": min_cost,
                    "path": min_path
                }

                self.log.debug(
                    f"Path between {self.terminals[terminal_i]} and {self.terminals[terminal_j]} ({min_path}) - cost ({min_cost})")

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
        while len(conn_components) > 1:
            comp1 = conn_components[0]
            comp2 = conn_components[1]
            for j in range(0, len(comp1)):
                for k in range(0, len(comp2)):
                    if self.graph.has_edge(comp1[j], comp2[k]):
                        self.steiner_tree.add_edge(comp1[j], comp2[k])
                        break
            conn_components = list(comp.connected_components(self.steiner_tree))

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
