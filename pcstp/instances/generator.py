import random
import numpy as np
import networkx as nx


def generate_random_steiner(
    num_nodes: int = 10,
    num_edges: int = 10,
    max_node_degree: int = 10,
    min_prize: int = 0,
    max_prize: int = 100,
    num_terminals: int = 5,
    min_edge_cost: int = 1,
    max_edge_cost: int = 10,
    cost_as_length: bool = False,
    max_iter: int = 100,
    seed: int = None
):
    """
    Produces a prize-collection steiner instance according to the given parameters.

    Args:
        num_nodes (int): number of instance nodes
        num_edges (int): maximum number of edges of the instance
        max_node_degree (int): maximum number of connections a node can make (degree)
        min_prize (int): minimum award for terminal nodes
        max_prize (int): maximum award for terminal nodes
        num_terminals (int): number of terminals in the instance
        min_edge_cost (int): minimum edge cost
        max_edge_cost(int): maximum edge cost
        cost_as_length (boolean): if true, edge cost is defined by arc length, calculated as the Euclidean distance
            between the points.
        max_iter (int): internal parameter to control maximum number of attempts to generate arcs according to the
            rules of the algorithm (avoid infinite loops)
        seed (Optional[int]): Seed used for reproductibility
        Returns:
            An instance of the prize collecting steiner problem in the form of a Graph (NetworkX) and a tuple containing
            the list of nodes (nodes), edges (edges), the position of nodes (position_matrix), the cost of edges (edges_cost),
            the list of selected terminals (terminals) and the prizes for each selected terminal (prizes).
    """
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    # Inicializar Arrays
    # Contagem de Grau dos nós
    degree_count = np.zeros(num_nodes)

    # Probabilidade Inicial do Nó ser escolhido (1 - Grau) / Soma((1 - Grau))
    nodes = list(range(0, num_nodes))
    node_prob = (1 - degree_count) / (1 - degree_count).sum()

    # Lista de Arestas
    edges = []
    edges_cost = []

    # Gerar num_nodes pontos aleatórios no R²(0, 100)
    position_matrix = np.random.rand(2, num_nodes) * 100

    # Selecionar aleatoriamente num_terminals nós para serem terminais
    terminals = set()

    # while len(terminals) < num_terminals:
    #     terminals.add(random.randint(0, num_nodes))
    terminals = np.random.choice(list(range(0, num_nodes)), size=num_terminals, replace=False)
    print("terminals: ", terminals.shape)

    # Gerar prêmios aleatórios para os nós terminais
    prizes = np.random.randint(min_prize, max_prize+1, size=num_terminals)
    print("prizes: ", prizes.shape)

    # Geração das Arestas
    for e in range(num_edges):

        i = 0

        # Escolhe dois nós aleatoriamente com base na probabilidade de escolha
        u, v = np.random.choice(list(range(0, num_nodes)), 2, p=node_prob)

        # Verifica se os nós escolhidos são os mesmos, enquanto forem, v
        # será sorteado novamente
        while u == v or (u, v) in edges or (v, u) in edges or i < max_iter:
            u, v = np.random.choice(list(range(0, num_nodes)), 2, p=node_prob)
            i += 1

        # Garantidos u != v, deg(u) < max_degree, deg(v) < max_degree
        if (u, v) not in edges and (v, u) not in edges:

            # Adicionar as listas
            edges.append((u, v))

            # Adiciona à contagem de grau
            degree_count[u] += 1
            degree_count[v] += 1

            # Recalcula probabilidades
            node_prob = (1 - degree_count / degree_count.sum())
            node_prob = node_prob / node_prob.sum()

    # Gerar custos para arestas
    if cost_as_length:
        for edge in edges:
            u = np.array((position_matrix[0][edge[0]]),
                         position_matrix[1][edge[0]])
            v = np.array((position_matrix[0][edge[1]]),
                         position_matrix[1][edge[1]])

            euclidean_distance = np.linalg.norm(u-v)

            edges_cost.append(euclidean_distance)

    else:
        # Gerar custo aleatoriamente
        edges_cost = np.random.randint(min_edge_cost, max_edge_cost + 1, size=len(edges))

    # Criar grafo
    G = nx.Graph()

    # Cria Nós
    for node in nodes:
        # Se o nó é terminal, adiciona o custo
        if node in terminals:
            terminal_idx = np.where(terminals == node)
            G.add_node(
                node,
                pos=(position_matrix[0][node], position_matrix[1][node]),
                terminal=True,
                prize=prizes[terminal_idx][0]
            )
        else:
            G.add_node(
                node,
                pos=(position_matrix[0][node], position_matrix[1][node]),
                terminal=False,
                prize=0
            )

    # Cria arestas
    for j, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], cost=edges_cost[j])

    # Verifica se o grafo está completamente conectado
    graph_connected_components = sorted(nx.connected_components(G))

    # Se houver mais de um subgrafo desconectado
    if len(graph_connected_components) > 1:
        for subgraph in (graph_connected_components[1:]):

            # Seleciona o primeiro nó do subgrafo
            u = list(subgraph)[0]

            # Seleciona um nó qualquer do maior componente
            v = list(graph_connected_components[0])[random.randint(0, len(graph_connected_components[0]))]

            # Define aresta e custo
            edge = (u, v)

            if cost_as_length == True:
                cost = np.linalg.norm(
                    np.array([position_matrix[0][u], position_matrix[1][u]]) -
                    np.array(position_matrix[0][v], position_matrix[1][v])
                )
            else:
                cost = random.randint(min_edge_cost, max_edge_cost + 1)

            # Adiciona à lista fixa
            edges.append(edge)
            np.append(edges_cost, cost)

            # Cria aresta
            G.add_edge(u, v, cost=cost)

    return G, (nodes, edges, position_matrix, edges_cost, terminals, prizes)
