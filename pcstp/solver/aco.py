"""Module that implements Ant Colony Optmization Solver for Prize-Collecting Steiner Tree Problem"""

import random
from typing import List, Tuple
from charset_normalizer import logging

import networkx as nx
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
            pheromone_per_iteration: float = 1,
            pheromone_initialization_strategy: str = 'same',
            **kwargs):
        """
        Prize Collecting Steiner Tree Solver based on Ant Colony Optmization 

        Args:
            graph (nx.Graph): [description]
            terminals (set): [description]
            iterations (int, optional): [description]. Defaults to 10.
            num_ants (int, optional): [description]. Defaults to 10.
            evaporation_rate (float, optional): [description]. Defaults to 0.1.
            alpha (float, optional): [description]. Defaults to 1.0.
            beta (float, optional): [description]. Defaults to 0.0.
            beta_evaporation_rate (float, optional): [description]. Defaults to 0.
            initial_pheromone (float, optional): [description]. Defaults to 1.
            pheromone_per_iteration (float, optional): [description]. Defaults to 1.
            pheromone_initialization_strategy (str, optional): [description]. Defaults to 'same'.
        """
        super().__init__(graph, terminals, **kwargs)

        self.iterations = iterations
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.pheromone_per_iteration = pheromone_per_iteration
        self.pheromone_initialization_strategy = pheromone_initialization_strategy

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

        logging.debug("Initializing pheromones...")

        if self.pheromone_initialization_strategy == 'same':
            initial_pheromone = self.initial_pheromone
        elif self.pheromone_initialization_strategy == 'heuristic':
            # TODO: Adds a initialization scheme based on heuristic
            initial_pheromone = {}

        nx.set_edge_attributes(self.graph, initial_pheromone, name='pheromone')

        self.num_nodes = len(self.nodes)

    def trace_pheromone(self, route: List[int], route_cost: float):
        """For a given route and its cost, deposit pheromone on all edges

        Args:
            route (List[int]): [description]
            route_cost (float): [description]
        """
        pheromone_deposit = self.pheromone_per_iteration / route_cost

        for nodes in range(self.num_nodes):
            idx1 = route[nodes]
            idx2 = route[(nodes+1) % len(route)]
            self.graph[idx1][idx2]['pheromone'] += pheromone_deposit

    def evaporate(self):
        """
        Applies evaporation strategy to all edges pheromones
        """
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                self.graph[i][j]['pheromone'] *= (1 - self.evaporation_rate)

    def _solve(self) -> Tuple[nx.Graph, int]:
        """Solve Prize-Collecting Steiner Tree based on Ant Colony Optimization

        Returns:
            Tuple[nx.Graph, int]: Returns the Steiner Tree and its cost
        """

        for ant in self.ants:
            ant.begin()

        for generation in range(self.iterations):
            # TODO: Is this the best strategy?
            # While the first ant hasn't reached the end, makes all turn
            while not self.ants[0].has_reached_end():
                # For each ant, turn
                for ant in self.ants:
                    ant.turn()

            self.evaporate()

            for ant in self.ants:
                ant.begin()

            self.log.debug("Generation: %s - Cost: %s", generation, 0)


class Ant():
    def __init__(self, antcolony: AntColony, name: str):
        """
        Ant is an agent of Ant Colony Optimization Technique

        Args:
            antcolony (AntColony): AntColony where the current ant lives
            name (str): Name of the current ant
        """
        self.antcolony = antcolony
        self.name = name

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

        The ant reach its end when all nodes have been visited
        """
        return self.has_visited_all_nodes

    def turn(self):
        """
        If the ant hasn't reached the end, chooses the next node.
        """
        if not self.has_visited_all_nodes:
            self.move()

            # If the Ant has visited all nodes
            if len(self.route) == self.antcolony.dimension:
                self.has_visited_all_nodes = True
                self.antcolony.trace_pheromone(self.antcolony._get_path_cost(self.route), self.route)

    def move(self):
        """
        Chooses next node to visit
        """
        for node in range(1, self.antcolony.dimension + 1):
            if self.current_node == node or node in self.route:
                continue

            if self.antcolony.edges[(node, self.current_node)] == 0:
                self.route.append(node)
                self.current_node = node

        # TODO:
