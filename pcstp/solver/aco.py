import random


class AntColony():
    def __init__(self, G):
        self.G = G
        self.nodes = G.nodes
        self.pheromone = {}
        self.edges = {}

        self.weight_max = 0
        self.weight_min = 9999999999

        self.dimension = len(self.nodes)

        for i in range(self.dimension):
            self.pheromone[(i, i)] = 0
            self.edges[(i, i)] = 0
            for j in range(i+1, self.dimension):
                dist = ((self.nodes[i][1] - self.nodes[j][1]) ** 2 + (self.nodes[i][2] - self.nodes[j][2]) ** 2) ** 0.5
                self.edges[(i, j)] = dist
                self.edges[(j, i)] = dist

                if self.weight_max < dist:
                    self.weight_max = dist
                if self.weight_min > dist:
                    self.weight_min = dist

                self.pheromone[(i, j)] = INITIAL_PHEROMONE
                self.pheromone[(j, i)] = INITIAL_PHEROMONE

    def trace_pheromone(self, L, route):
        deposit = ESTIMATED_SHORTEST_TOUR / L
        for idx in range(self.dimension):
            idx1 = route[idx] - 1
            idx2 = route[(idx+1) % len(route)] - 1

            self.pheromone[(idx1, idx2)] += deposit
            self.pheromone[(idx2, idx1)] += deposit

    def evaporate(self):
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                self.pheromone[(i, j)] *= (1 - EVAPORATION_RATE)
                self.pheromone[(j, i)] *= (1 - EVAPORATION_RATE)


class Ant():
    def __init__(self, antcolony: AntColony, name: str):
        self.antcolony = antcolony
        self.name = name

    def begin(self):
        self.current = random.randint(1, self.antcolony.dimension)
        self.route = [self.current]
        self.has_reached_end = False

    def end(self):
        return self.has_reached_end

    def turn(self):
        if not self.has_reached_end:
            self.move()
            if len(self.route) == self.antcolony.dimension:
                self.has_reached_end = True
                self.antcolony.trace_pheromone(self.route_cost(self.antcolony.nodes), self.route)

    def move(self):
        prob_divider = 0
        for i in range(1, self.antcolony.dimension + 1):
            if self.current == i or i in self.route:
                continue

            if self.antcolony.edges[(i-1, self.current-1)] == 0:
                self.route.append(i)
                self.current = i

                return

            dist = self.antcolony.edges[(i-1, self.current-1)]
            eta = (self.antcolony.weight_max - self.antcolony.weight_min) / (dist - self.antcolony.weight_min + 1)
            if self.testFlag:
                eta = 1
            tau = self.antcolony.pheromone[(i-1, self.current-1)]

            prob_divider += (eta**WEIGHT_LENGTH) * (tau**WEIGHT_PHEROMONE)

        prob_list = []
        prob_index = []
        for i in range(1, self.antcolony.dimension + 1):
            if self.current == i or i in self.route:
                continue

            dist = self.antcolony.edges[(i-1, self.current-1)]
            eta = (self.antcolony.weight_max - self.antcolony.weight_min) / (dist - self.antcolony.weight_min + 1)
            if self.testFlag:
                eta = 1
            tau = self.antcolony.pheromone[(i-1, self.current-1)]

            prob = (eta**WEIGHT_LENGTH) * (tau**WEIGHT_PHEROMONE)
            prob_list.append(prob / prob_divider)
            prob_index.append(i)

        prob = random.random()

        found = -1
        for i in range(len(prob_list)):
            prob -= prob_list[i]
            if prob < 0:
                found = prob_index[i]
                break

        self.route.append(found)
        self.current = found

    def route_cost(self, nodes):
        total = 0
        for idx in range(len(self.route)):
            n1 = nodes[self.route[idx] - 1]
            n2 = nodes[self.route[(idx+1) % len(self.route)] - 1]

            total += ((n1[1] - n2[1])**2 + (n1[2] - n2[2])**2)**0.5
        return total
