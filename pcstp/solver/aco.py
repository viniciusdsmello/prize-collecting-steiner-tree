"""Module that implements Ant Colony Optmization Solver for Prize-Collecting Steiner Tree Problem"""

import time
import random

import numpy as np
import networkx as nx
import networkx.algorithms.components as comp

from typing import List, Tuple
from networkx.utils import pairwise
from pcstp.solver.base import BaseSolver


class AntColony(BaseSolver):
    def __init__(
            self,
            graph: nx.Graph,
            terminals: set,
            iterations: int = 10,
            num_ants: int = 10,
            evaporation_rate: float = 0.1,
            alpha: float = 1.0,
            beta: float = 0.0,
            beta_evaporation_rate: float = 0,
            initial_pheromone: float = 1,
            pheromone_amount: float = 1,
            pheromone_initialization_strategy: str = 'same_value',
            choose_best: float = 0.1,
            early_stopping: int = None,
            **kwargs):
        """
        Prize Collecting Steiner Tree Solver based on Ant Colony Optmization

        Args:
            graph (nx.Graph): Instance Graph
            terminals (set): List of Terminals
            iterations (int, optional): Number of iterations. Defaults to 10.
            num_ants (int, optional): Number of ants. Defaults to 10.
            evaporation_rate (float, optional): Pheromone evaporation rate. Defaults to 0.1.
            alpha (float, optional): [description]. Defaults to 1.0.
            beta (float, optional): [description]. Defaults to 0.0.
            beta_evaporation_rate (float, optional): Beta evaporation rate. Defaults to 0.
            initial_pheromone (float, optional): Initial pheromone value. Defaults to 1.
            pheromone_amount (float, optional): Pheromone amout deposited by each ant. Defaults to 1.
            pheromone_initialization_strategy (str, optional): Pheromone initialization strategy.
                When 'same_value' is passed, all edges are initialized with `initial_pheromone`.
                When 'heuristic' is passed, a greedy heuristic is used to initialize the edges with initial pheromone.
                Defaults to 'same_value'.
            choose_best (float, optional): Indicates at which probability the best path will be take by each ant. Defaults to 0.1.
        """
        super().__init__(graph, terminals, **kwargs)

        assert evaporation_rate > 0, 'evaporation_rate must be greater than 0'
        assert evaporation_rate <= 1, 'evaporation_rate must be less or equal than 1'

        self.iterations = iterations
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.pheromone_amount = pheromone_amount
        self.pheromone_initialization_strategy = pheromone_initialization_strategy
        self.choose_best = choose_best

        if early_stopping:
            self.early_stopping = early_stopping
        else:
            self.early_stopping = self.iterations

        self.ants: List[Ant] = []

        self.__initialize()

    def __initialize(self):
        """
        Initializes AntColony and Ants
        """
        self.log.debug("Initializing ants...")
        self.ants = [
            Ant(self, ant) for ant in range(self.num_ants)
        ]

        self.log.debug("Initializing pheromones (strategy: %s)...", self.pheromone_initialization_strategy)

        if self.pheromone_initialization_strategy == 'same_value':
            initial_pheromone = self.initial_pheromone
        elif self.pheromone_initialization_strategy == 'heuristic':
            # TODO: Adds a initialization scheme
            initial_pheromone = {}

        nx.set_edge_attributes(self.graph, initial_pheromone, name='pheromone')

        self.num_nodes = len(self.graph.nodes)

    def trace_pheromone(self, route: List[int], route_cost: float):
        """For a given route and its cost, deposit pheromone on all edges

        Args:
            route (List[int]): Ant route
            route_cost (float): Ant's route cost
        """
        pheromone_deposit = self.pheromone_amount / route_cost

        updated_edges = []
        for edge in self.graph.edges:
            if edge not in updated_edges:
                i, j = edge
                self.log.debug("Evaporating pheromone of edge (%s, %s) ...", i, j)
                self.graph[i][j]['pheromone'] += pheromone_deposit
                updated_edges.append(edge)

    def evaporate(self):
        """
        Applies evaporation strategy to all edges pheromones and also to beta
        """
        updated_edges = []
        for edge in self.graph.edges:
            if edge not in updated_edges:
                i, j = edge
                self.log.debug("Evaporating pheromone of edge (%s, %s) ...", i, j)
                self.graph[i][j]['pheromone'] *= (1 - self.evaporation_rate)
                updated_edges.append(edge)

        self.beta *= (1 - self.beta_evaporation_rate)

    def _evaluate_solution(self) -> Tuple[nx.Graph, float]:
        self.log.debug("Evaluating solution...")

        # Reset solution
        if len(self.steiner_tree.nodes) > 0:
            self.steiner_tree = nx.Graph()
            self.steiner_cost = None

        all_routes = []
        for terminal in self.terminals:
            evaluation_ant = Ant(self, 'evaluation_ant', update_pheromones=False)
            evaluation_ant.begin(terminal)

            while not evaluation_ant.has_reached_end():
                evaluation_ant.turn()

            all_routes.append(evaluation_ant.route)

        for route in all_routes:
            # Transform route into steiner_graph
            nx.add_path(self.steiner_tree, route)

            # If all terminals are already connected, break the loop
            if self.is_all_terminals_connected():
                break

        # Adds all edges from graph to steiner tree solution
        conn_components = list(comp.connected_components(self.steiner_tree))
        while len(conn_components) > 1:
            comp1 = list(conn_components[0])
            comp2 = list(conn_components[1])
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

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree based on Ant Colony Optimization

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        best_cost: float = float('inf')
        best_route: List[int] = []
        best_solution: nx.Graph = None
        best_iteration: int = None
        iterations_without_improvement = 0

        for ant in self.ants:
            ant.begin()

        for iteration in range(self.iterations):
            self._iteration_start_time = time.time()
            # TODO: Is this the best strategy?
            # While the first ant hasn't reached the end, makes all turn
            while not self.ants[0].has_reached_end():
                # For each ant, move based on probability and updates pheromones
                for ant in self.ants:
                    ant.turn()

            # Applies evaporation strategy
            self.evaporate()

            solution, cost = self._evaluate_solution()

            self.log.debug("Cost: %s", cost)
            if cost < best_cost:
                self.log.debug("New best cost found!")
                best_iteration = iteration
                best_cost = cost
                best_solution = solution
            else:
                iterations_without_improvement += 1

            self.log.debug("Iteration: %s - Cost: %s", iteration, cost)

            if iterations_without_improvement > self.early_stopping:
                self.log.debug("Early Stopping: %s", iteration)
                break

            # Set a new begin for each ant
            for ant in self.ants:
                ant.begin()

        self.log.debug("Best Iteration: %s - Best Cost: %s", best_iteration, best_cost)

        self.steiner_tree, self.steiner_cost = (best_solution, best_cost)

        return self.steiner_tree, self.steiner_cost

    def _get_neighbors(self, node: int) -> List[int]:
        return list(self.graph.neighbors(node))


class Ant():
    def __init__(
            self,
            antcolony: AntColony,
            name: str,
            update_pheromones: bool = True
    ):
        """
        Ant is an agent of Ant Colony Optimization Technique

        Args:
            antcolony (AntColony): AntColony where the current ant lives
            name (str): Name of the current ant
            update_pheromones (bool, optional): If True the ant can deposit pheromones. Defaults to True
        """
        self.antcolony = antcolony
        self.name = name
        self.update_pheromones = update_pheromones

        self.log = self.antcolony.log

        self.route: List[int] = []
        self.has_visited_all_nodes: bool = False

        self._current_node: int = None
        self._previous_node: int = None

    @property
    def current_node(self):
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        self._current_node = value
        self.candidate_neighbors = self.antcolony._get_neighbors(self.current_node)

    def begin(self, current_node: int = None):
        """
        # TODO: update docstring
        Chooses randomly among terminals the node where the ant will begin its route
        """
        if current_node:
            self.current_node = current_node
        else:
            self.current_node = random.choice(self.antcolony.terminals)
        self.log.debug("Ant %s is beginnig from node %s", self.name, self.current_node)

        # Adds the current node to ants path
        self.route = [self.current_node]

        self.has_visited_all_nodes = False
        self.has_reached_leaf_node = False

    def has_reached_end(self):
        """
        Indicates whether the ant has reached the end.
        """
        reached_end = False

        current_neighbors = self.antcolony._get_neighbors(self.current_node)
        all_terminals_in_route = set(self.antcolony.terminals).issubset(set(self.route))
        no_neighbors_to_visit = len(current_neighbors) == 0

        if self.has_visited_all_nodes or all_terminals_in_route or no_neighbors_to_visit or self.has_reached_leaf_node:
            reached_end = True

        return reached_end

    def turn(self):
        """
        If the ant hasn't reached the end, chooses the next node.

        The ant reach its end when all nodes have been visited or all terminals have been visited
        """
        if not self.has_reached_end():
            self.move()

            # The ant only deposits pheromone when all nodes were visited
            if len(self.route) == len(self.antcolony.graph.nodes):
                self.has_visited_all_nodes = True

            if self.has_reached_leaf_node:
                if self.update_pheromones:
                    self.log.debug("Ant %s is depositing pheromone...", self.name)
                    self.antcolony.trace_pheromone(self.route, self.antcolony._get_path_cost(self.route))

    def move(self):
        """
        Chooses next node to visit based on probabilities. 

        If p < p_choose_best, then the best path is chosen, otherwise it is selected
        from a probability distribution weighted by the pheromone.
        """
        neighbors_transition_probability = np.zeros(shape=(len(self.candidate_neighbors)))

        def get_tau_and_eta(node: int) -> Tuple[float, float]:
            distance = self.antcolony.graph[self.current_node][node]['cost']
            prize = self.antcolony.graph.nodes[node]['prize']
            pheromone = self.antcolony.graph[self.current_node][node]['pheromone']

            eta = (1 + prize) / (1 + distance)
            tau = pheromone

            return tau, eta

        for i, node in enumerate(self.candidate_neighbors):
            # Skips previous node visited
            if len(self.route) > 1:
                if node == self.route[-2]:
                    self.log.debug("Candidate node %s was already visited, setting it's probability to 0.0", node)
                    continue
            if len(self.antcolony._get_neighbors(node)) == 1 and node not in self.antcolony.terminals:
                self.log.debug("Candidate node %s is a non-terminal leaf, setting it's probability to 0.0", node)
                neighbors_transition_probability[i] = 0
            elif len(self.antcolony._get_neighbors(node)) > 1 and node == self.route[-1]:
                self.log.debug("Candidate node is the previous node, setting it's probability to 0.0", self.route[-1])
                neighbors_transition_probability[i] = 0
            else:
                eta, tau = get_tau_and_eta(node)
                neighbors_transition_probability[i] = (tau**self.antcolony.alpha) * (eta**self.antcolony.beta)

        if np.all(neighbors_transition_probability == 0):
            self.log.debug("Ant %s has reached a leaf", self.name)
            self.log.debug("Ant %s route: %s", self.name, self.route)
            self.has_reached_leaf_node = True

            return

        neighbors_transition_probability = neighbors_transition_probability / np.sum(neighbors_transition_probability)
        self.log.debug('transition_probability (%s) %s', self.candidate_neighbors, neighbors_transition_probability)

        # Given a transition probability matrix, choose which node is going to be visited by the ant
        if np.random.random() < self.antcolony.choose_best:
            next_node_index = np.argmax(neighbors_transition_probability)
            next_node = self.candidate_neighbors[next_node_index]
        else:
            self.log.debug("Choosing one of %s", self.candidate_neighbors)
            next_node = np.random.choice(self.candidate_neighbors, 1, p=neighbors_transition_probability)

        next_node = int(next_node)
        self.route.append(next_node)
        self.log.info("Ant %s move (%s -> %s)", self.name, self.current_node, next_node)
        self._previous_node = self.current_node
        self.current_node = next_node
