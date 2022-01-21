from typing import Tuple, Set
import networkx as nx

def preprocessing(graph: nx.Graph, terminals: set) -> Tuple[nx.Graph, Set[int]]:
    """Pre-Processing Function applied to Instace Graph in order to reduce complexity.

    It removes nodes with degree less or equal that are not terminal nodes

    Args:
        graph (nx.Graph): Instance graph
        terminals (set): Set of terminal nodes

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
        print(f'Iteration: {iteration} - Removing nodes: {nodes_to_remove}')
        final_graph.remove_nodes_from(nodes_to_remove)
        nodes_to_remove = get_nodes_to_remove(final_graph, terminals)
        iteration += 1

    return final_graph, terminals