from tabnanny import verbose
from typing import Set, Tuple

import networkx as nx
import networkx.algorithms.components as comp


def preprocessing(graph: nx.Graph, terminals: set, verbose: bool = False) -> Tuple[nx.Graph, Set[int]]:
    """Pre-Processing Function applied to Instace Graph in order to reduce complexity.

    It removes nodes with degree less or equal that are not terminal nodes

    Args:
        graph (nx.Graph): Instance graph
        terminals (set): Set of terminal nodes
        verbose (bool, optional): When True prints preprocessing iterations. Defaults to False

    Returns:
        Tuple[nx.Graph, Set[int]]: Returns a processed version of Instance Graph and its terminals
    """
    final_graph = graph.copy()

    def get_nodes_to_remove(graph, terminals):
        return [
            int(node) for node, degree in dict(graph.degree()).items()
            if degree <= 1 and node not in terminals
        ]
    nodes_to_remove = get_nodes_to_remove(graph, terminals)
    iteration = 1
    while len(nodes_to_remove) > 0:
        if verbose:
            print(f'Iteration: {iteration} - Removing nodes: {nodes_to_remove}')
        final_graph.remove_nodes_from(nodes_to_remove)
        nodes_to_remove = get_nodes_to_remove(final_graph, terminals)
        iteration += 1

    conn_components = list(comp.connected_components(final_graph))
    for component in conn_components:
        has_terminals = False
        for terminal in terminals:
            if terminal in component:
                has_terminals = True
                break
        if not has_terminals:
            for node in component:
                final_graph.remove_node(node)

    conn_components = list(comp.connected_components(final_graph))
    if len(conn_components) > 1:
        conn_components.sort(key=lambda x: len(set(x).intersection(set(terminals))))
        final_graph = final_graph.subgraph(conn_components[-1])

    return final_graph, terminals
