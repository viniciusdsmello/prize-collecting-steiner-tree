"""Module that implements Ant Colony Optmization Solver for Prize-Collecting Steiner Tree Problem"""

import time
import random

import numpy as np
import networkx as nx
import networkx.algorithms.components as comp

from typing import List, Tuple
from networkx.utils import pairwise
from pcstp.solver.base import BaseSolver
from pcstp.utils.graph import preprocessing


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
        pheromone_deposit_strategy: str = 'traditional',
        pheromone_initialization_strategy: str = 'same_value',
        choose_best: float = 0.1,
        early_stopping: int = None,
        seed: int = 100,
        normalize_distance_prize: bool = False,
        allow_edge_perturbation: bool = False,
        ant_max_moves: int = 100,
        **kwargs
    ):
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
            pheromone_deposit_strategy (str, optional): Pheromone deposit strategy.
                When its value equals to 'traditional', each ant can deposit its pheromone.
                When its value equals to 'best_route', only the best ant deposits pheromone.
                Defaults to 'traditional'.
            pheromone_initialization_strategy (str, optional): Pheromone initialization strategy.
                When 'same_value' is passed, all edges are initialized with `initial_pheromone`.
                When 'heuristic' is passed, a greedy heuristic is used to initialize the edges with initial pheromone.
                It then finds the lowest cost path between all terminals pairs, then sets initial_pheromone on its edges.
                The remaining edges are initialized with initial_pheromone / 2.
                Defaults to 'same_value'.
            choose_best (float, optional): Indicates at which probability the best path will be take by each ant. Defaults to 0.1.
            early_stopping (int, optional): Indicates how many iterations without improvement should be tolerated.
            seed (int, optional): Seed used for experiments reproductibility
            normalize_distance_prize (bool, optional): If true, applies MinMax normalization to edges cost and nodes prizes.
                Defaults to 'False'.
            allow_edge_perturbation (bool, optional): If true, adds a uniform distributed perturbation to edges cost.
                Defaults to 'False'.
            ant_max_moves (bool, opttional): Set the max moves an ant can make without reaching all terminals. Defaults to 100.
        """
        super().__init__(graph, terminals, **kwargs)

        assert num_ants > 0, 'Invalid number of ants'
        assert iterations > 0, 'Invalid number of iterations'

        assert evaporation_rate > 0, 'evaporation_rate must be greater than 0'
        assert evaporation_rate <= 1, 'evaporation_rate must be less or equal than 1'

        assert pheromone_deposit_strategy in (
            'traditional', 'best_route'), 'Invalid value for pheromone_deposit_strategy'
        assert pheromone_initialization_strategy in (
            'same_value', 'heuristic'), 'Invalid value for pheromone_deposit_strategy'

        self.iterations = iterations
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.pheromone_amount = pheromone_amount
        self.pheromone_deposit_strategy = pheromone_deposit_strategy
        self.pheromone_initialization_strategy = pheromone_initialization_strategy
        self.choose_best = choose_best
        self.normalize_distance_prize = normalize_distance_prize
        self.allow_edge_perturbation = allow_edge_perturbation
        self.ant_max_moves = ant_max_moves

        if early_stopping:
            self.early_stopping = early_stopping
        else:
            self.early_stopping = self.iterations

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.ants: List[Ant] = []

        self.history: List[float] = []

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
            initial_pheromone = {}
            _all_terminals_path = []
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
                    _all_terminals_path.append(path)

            # After finding all paths between terminals
            # Sort them based on cost
            _all_terminals_path.sort(key=lambda path: path['cost'])

            best_edges = []

            for terminal_path in _all_terminals_path:
                self.log.debug("Depositing initial_pheromone to path %s", terminal_path)
                best_edges.append(list(set(pairwise(terminal_path['path']))))

            initial_pheromone = dict(
                [
                    edge,
                    self.initial_pheromone if edge in best_edges else self.initial_pheromone * 0.5
                ]
                for edge in self.graph.edges
            )

        nx.set_edge_attributes(self.graph, initial_pheromone, name='pheromone')

        self.num_nodes = len(self.graph.nodes)

        self.best_cost: float = float('inf')
        self.best_solution: nx.Graph = None
        self.best_iteration: int = None
        self.best_route: List[int] = None

    def trace_pheromone(self, route: List[int], route_cost: float):
        """For a given route and its cost, deposit pheromone on all edges

        Args:
            route (List[int]): Ant route
            route_cost (float): Ant's route cost
        """
        pheromone_deposit = self.pheromone_amount / route_cost

        updated_edges = []
        route_edges = pairwise(route)
        for edge in route_edges:
            if edge not in updated_edges:
                i, j = edge
                self.log.debug("Depositing pheromone on edge (%s, %s): %s...", i, j, pheromone_deposit)
                self.graph.edges[i, j]['pheromone'] += pheromone_deposit
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

        # Computes Steiner Tree for each evaluation ant starting on each terminal
        solutions = []
        for terminal in set(self.terminals).intersection(self.graph):
            evaluation_ant = Ant(self, f'evaluation_ant_terminal_{terminal}', update_pheromones=False)
            evaluation_ant.begin(terminal)

            while not evaluation_ant.has_reached_end():
                evaluation_ant.turn()

            tree = nx.Graph()

            # Transform route into steiner_graph
            nx.add_path(tree, evaluation_ant.route)

            # While the steiner tree graph has any cycle, then remove one edge
            tree = self._process_cycles(steiner_tree=tree)
            tree, _ = preprocessing(tree, set(self.terminals), verbose=False)

            solutions.append(
                {
                    'tree': tree,
                    'cost': self._get_steiner_cost(tree),
                    'route': evaluation_ant.route
                }
            )

        solutions.sort(key=lambda path: path['cost'])

        best_steiner_tree = solutions[0]['tree']
        best_steiner_cost = solutions[0]['cost']
        best_ants_route = solutions[0]['route']

        self.log.debug('Best Route: %s', best_ants_route)

        return best_steiner_tree, best_steiner_cost, best_ants_route

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree based on Ant Colony Optimization

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """
        iterations_without_improvement = 0

        for ant in self.ants:
            ant.begin()

        for iteration in range(self.iterations):
            self.log.debug('Iteration %s', iteration)
            self._iteration_start_time = time.time()

            while all([not ant.has_reached_end() for ant in self.ants]):
                # For each ant, move based on probability and updates pheromones
                for ant in self.ants:
                    ant.turn()

            if self.pheromone_deposit_strategy == 'best_route':
                paths = [ant.route for ant in self.ants]
                min_path, min_cost = self._get_least_cost_path(paths)
                path_index = paths.index(min_path)
                self.log.debug('Depositing pheromone on ant %s route...', self.ants[path_index].name)
                self.trace_pheromone(min_path, min_cost)

            # Applies evaporation strategy
            self.evaporate()

            solution, cost, route = self._evaluate_solution()

            self.log.debug("Cost: %s", cost)
            if cost < self.best_cost:
                self.log.debug("New best cost found!")
                self.best_iteration = iteration
                self.best_cost = cost
                self.best_solution = solution
                self.best_route = route
            else:
                iterations_without_improvement += 1

            self.history.append(cost)

            self.log.debug("Iteration: %s - Cost: %s", iteration, cost)

            if iterations_without_improvement > self.early_stopping:
                self.log.info("Early Stopping: %s", iteration)
                break

            # Set a new begin for each ant
            for ant in self.ants:
                ant.begin()

        self.log.info("Best Iteration: %s - Best Cost: %s",
                      self.best_iteration, self.best_cost)
        self.log.debug("Best ants route: %s", self.best_route)

        self.steiner_tree, self.steiner_cost = (self.best_solution, self.best_cost)

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
        self.has_visited_all_terminals: bool = False

        self._current_node: int = None

        self.max_prize = max(list(nx.get_node_attributes(self.antcolony.graph, 'prize').values()))
        self.min_prize = min(list(nx.get_node_attributes(self.antcolony.graph, 'prize').values()))

        self.max_distance = max(list(nx.get_edge_attributes(self.antcolony.graph, 'cost').values()))
        self.min_distance = min(list(nx.get_edge_attributes(self.antcolony.graph, 'cost').values()))

    @property
    def current_node(self):
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        self._current_node = value
        self.candidate_neighbors = self.antcolony._get_neighbors(self.current_node)

    def begin(self, current_node: int = None):
        """
        Sets ants initial route by setting its first node and route.

        Args:
            current_node (int, optional): When current_node is passed it is used as the start of Ant,
            when it is None, a terminal is chosen randomly.
        """
        if current_node:
            self.current_node = current_node
        else:
            possible_choices = list(set(self.antcolony.terminals).intersection(self.antcolony.graph))
            self.current_node = random.choice(possible_choices)
        self.log.debug("Ant %s is beginnig from node %s", self.name, self.current_node)

        # Adds the current node to ants path
        self.route = [self.current_node]

        self.has_visited_all_nodes = False
        self.has_visited_all_terminals = False

    def has_reached_end(self):
        """
        Indicates whether the ant has reached the end.
        """
        reached_end = False

        current_neighbors = self.antcolony._get_neighbors(self.current_node)
        self.has_visited_all_terminals = set(self.antcolony.terminals).issubset(set(self.route))
        no_neighbors_to_visit = len(current_neighbors) == 0

        if self.has_visited_all_nodes or self.has_visited_all_terminals or no_neighbors_to_visit or len(self.route) > self.antcolony.ant_max_moves:
            reached_end = True

        return reached_end

    def turn(self):
        """
        If the ant hasn't reached the end, chooses the next node.

        The ant reach its end when all nodes have been visited or all terminals have been visited
        """
        if not self.has_reached_end():
            self.move()

            if set(self.route) == set(self.antcolony.graph.nodes):
                self.has_visited_all_nodes = True

            if set(self.antcolony.terminals).issubset(set(self.route)):
                self.has_visited_all_terminals = True

            if self.has_visited_all_nodes or self.has_visited_all_terminals or len(self.route) > self.antcolony.ant_max_moves:
                if self.update_pheromones and self.antcolony.pheromone_deposit_strategy == 'traditional':
                    self.log.debug("Ant %s is depositing pheromone - route %s", self.name, self.route)
                    self.antcolony.trace_pheromone(self.route, self.antcolony._get_path_cost(self.route))

    def move(self):
        """
        Chooses next node to visit based on probabilities. 

        If p < p_choose_best, then the best path is chosen, otherwise it is selected
        from a probability distribution weighted by the pheromone.

        p_ij{k} = \frac{tau^alpha + eta^beta}{\sum tau^alpha + eta^beta}

        """
        neighbors_transition_probability = np.zeros(shape=(len(self.candidate_neighbors)))

        def get_tau_and_eta(node: int) -> Tuple[float, float]:
            """Given a node calculates the parameters for the probability transition

            Args:
                node (int): Neighbor node of the current node.

            Returns:
                Tuple[float, float]: Returns a tuple where the first value is tau and the second one eta.
            """
            perturbation = random.random() if self.antcolony.allow_edge_perturbation else 0

            # TODO: When value is equal to min, how it affects eta formula?
            def min_max_normalization(value, min, max):
                return (value - min) / (max - min) if max != min else value

            distance = self.antcolony.graph[self.current_node][node]['cost'] * (1 + perturbation)
            prize = self.antcolony.graph.nodes[node]['prize']

            if self.antcolony.normalize_distance_prize:
                distance = min_max_normalization(distance, self.max_distance, self.min_distance)
                prize = min_max_normalization(prize, self.min_prize, self.max_prize)

            tau = self.antcolony.graph[self.current_node][node]['pheromone']

            # Eta is inversely proportional to edge cost and directly proportional to the prize the node offers
            # TODO: Check eta's formula. Traditional formula for ACO is eta = 1/distance
            eta = (0.1 + prize)/(0.1 + distance)
            return tau, eta

        for i, node in enumerate(self.candidate_neighbors):
            if node in self.route and len(self.route) > 1:
                # # If candidate node is the previous node visited, skip it
                if node == self.route[-2]:
                    self.log.debug("Candidate node %s is in the recent history, setting it's probability to 0.0", node)
                    neighbors_transition_probability[i] = 0
                    continue

                # If candidate node is already on ant's route, but isn't the previous one give it's transition probability an attenuation
                attenuation = self.route[::-1].index(node)
                attenuation = pow(0.5, attenuation)
                tau, eta = get_tau_and_eta(node)

                # Visiting a node previously visited isn't prohibited, but it's discourage by using an attenuation factor
                neighbors_transition_probability[i] = (pow(tau, self.antcolony.alpha) * pow(eta, self.antcolony.beta)) * (1 - attenuation)

                self.log.debug("Candidate node %s was already visited, decreasing it's probability by %s", node, attenuation)
            elif len(self.antcolony._get_neighbors(node)) == 1 and node not in self.antcolony.terminals:
                self.log.debug("Candidate node %s is a non-terminal leaf, setting it's probability to 0.0", node)
                neighbors_transition_probability[i] = 0
            else:
                tau, eta = get_tau_and_eta(node)
                neighbors_transition_probability[i] = (pow(tau, self.antcolony.alpha) * pow(eta, self.antcolony.beta))

        # The ant has reached a leaf node, it should return
        if np.all(neighbors_transition_probability == 0):
            # self.has_reached_leaf_node = True
            next_node = self.route[-2]
            self.log.debug("Ant %s has reached a leaf. Route: %s. Returning to %s", self.name, self.route, next_node)
        else:
            neighbors_transition_probability = neighbors_transition_probability / np.sum(neighbors_transition_probability)
            self.log.debug('transition_probability (%s) %s', self.candidate_neighbors, neighbors_transition_probability)

            # Given a transition probability matrix, choose which node is going to be visited by the ant
            if np.random.random() < self.antcolony.choose_best:
                self.log.debug("Choosing the best node...")
                next_node_index = np.argmax(neighbors_transition_probability)
                next_node = self.candidate_neighbors[next_node_index]
            else:
                self.log.debug("Choosing one of %s", self.candidate_neighbors)
                next_node = np.random.choice(self.candidate_neighbors, 1, p=neighbors_transition_probability)

        next_node = int(next_node)
        self.route.append(next_node)
        self.log.debug("Ant %s move (%s -> %s)", self.name, self.current_node, next_node)
        self.current_node = next_node
