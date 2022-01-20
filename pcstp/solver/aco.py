"""Module that implements Ant Colony Optmization Solver for Prize-Collecting Steiner Tree Problem"""

import random
import time
from typing import List, Tuple

import networkx as nx
import networkx.algorithms.components as comp
import numpy as np
from charset_normalizer import logging
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
            pheromone_initialization_strategy: str = 'same',
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
                When 'same' is passed, all edges are initialized with `initial_pheromone`.
                When 'heuristic' is passed, a greedy heuristic is used to initialize the edges with initial pheromone.
                Defaults to 'same'.
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
        logging.debug("Initializing ants...")
        self.ants = [
            Ant(self, ant) for ant in range(self.num_ants)
        ]

        logging.debug("Initializing pheromones (strategy %s)...", self.pheromone_initialization_strategy)

        if self.pheromone_initialization_strategy == 'same':
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

        for nodes in range(self.num_nodes):
            idx1 = route[nodes]
            idx2 = route[(nodes+1) % len(route)]
            self.graph[idx1][idx2]['pheromone'] += pheromone_deposit

    def evaporate(self):
        """
        Applies evaporation strategy to all edges pheromones and also to beta
        """
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                self.graph[i][j]['pheromone'] *= (1 - self.evaporation_rate)

        self.beta *= (1 - self.beta_evaporation_rate)

    def _evaluate_route(self, route: List[int]) -> Tuple[nx.Graph, float]:
        evalutation_graph = nx.Graph()

        # Transform route into steiner_graph
        # TODO: Checks if add_path works with ant's routes
        nx.add_path(evalutation_graph, route)

        # Adds all edges from graph to steiner tree solution
        conn_components = list(comp.connected_components(evalutation_graph))
        while len(conn_components) > 1:
            comp1 = conn_components[0]
            comp2 = conn_components[1]
            for j in range(0, len(comp1)):
                for k in range(0, len(comp2)):
                    if self.graph.has_edge(comp1[j], comp2[k]):
                        evalutation_graph.add_edge(comp1[j], comp2[k])
                        break
            conn_components = list(comp.connected_components(evalutation_graph))

        cycle = nx.find_cycle(evalutation_graph)
        self.log.debug(f'Cycle found - {cycle}')

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree based on Ant Colony Optimization

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        best_cost: float = float('inf')
        best_route: List[int] = []
        best_solution: nx.Graph = None
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

            # Set a new begin for each ant
            for ant in self.ants:
                ant.begin()

            # Evaluates solution
            evaluation_ant = Ant(self, 'evaluation_ant', update_pheromones=False)

            solution, cost = self._evaluate_route(evaluation_ant.route)

            if cost < best_cost:
                best_cost = cost
                best_route = evaluation_ant.route
                best_solution = solution
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement > self.early_stopping:
                break

            self.log.debug("Generation: %s - Best Cost: %s - Best Route %s", iteration, best_cost, best_route)

        self.steiner_tree, self.steiner_cost = best_solution, best_cost

        return self.steiner_tree, self.steiner_cost


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

        self.route: List[int] = []
        self.has_visited_all_nodes: bool = False

    def begin(self):
        """
        Chooses randomly among terminals the node where the ant will begin its route
        """
        self.current_node = random.choice(self.antcolony.terminals)

        # Adds the current node to ants path
        self.route = [self.current_node]

        self.has_visited_all_nodes = False

    def has_reached_end(self):
        """
        Indicates whether the ant has reached the end.
        """
        return self.has_visited_all_nodes or set(self.antcolony.terminals).issubset(set(self.route))

    def turn(self):
        """
        If the ant hasn't reached the end, chooses the next node.

        The ant reach its end when all nodes have been visited or all terminals have been visited
        """
        if not self.has_reached_end():
            self.move()

            # TODO: Should pheromone deposits be made independently? Wouldn't it be better to rank solutions?
            if len(self.route) == len(self.antcolony.graph.nodes):
                self.has_visited_all_nodes = True

                if self.update_pheromones:
                    self.antcolony.trace_pheromone(self.antcolony._get_path_cost(self.route), self.route)

    def move(self):
        """
        Chooses next node to visit based on probabilities. 

        If p < p_choose_best, then the best path is chosen, otherwise it is selected
        from a probability distribution weighted by the pheromone.
        """
        current_neighbors = list(self.antcolony.graph.neighbors(self.current_node))
        transition_probability = np.zeros(shape=(len(current_neighbors)))

        def get_tau_and_eta(node: int) -> Tuple[float, float]:
            distance = self.antcolony.graph[self.current_node][node]['cost']
            eta = 1 / distance
            tau = self.antcolony.graph[self.current_node][node]['pheromone']

            return tau, eta

        for i, node in enumerate(current_neighbors):
            if node in self.route:
                continue

            # If the node has zero cost, move to it
            if self.antcolony.graph[self.current_node][node]['cost'] == 0 and self.antcolony.graph[node]['prize'] >= 0:
                self.route.append(node)
                self.current_node = node

                return

            eta, tau = get_tau_and_eta(node)
            transition_probability[i] = (tau**self.antcolony.alpha) * (eta**self.antcolony.beta)

        transition_probability = transition_probability / np.sum(transition_probability)

        # Given a transition probability vector, choose
        if np.random.random() < self.antcolony.choose_best:
            next_node = np.argmax(transition_probability)
        else:
            next_node = np.random.choice(current_neighbors, p=transition_probability)

        self.route.append(next_node)
        self.antcolony.log.debug("Ant %s is moving from %s to %s", self.name, self.current_node, next_node)
