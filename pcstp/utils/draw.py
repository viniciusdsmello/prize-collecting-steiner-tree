import networkx as nx
import matplotlib.pyplot as plt


def draw_steiner_graph(G: nx.Graph, node_color: str = 'blue', terminal_color: str = 'red'):
    """
    Function that receives a PCSTP Graph and draws its nodes and edges

    Args:
        G (nx.Graph): Prize-Collecting Steiner Tree Instance.
        node_color (str): Color used for nodes that are not terminals
        terminal_color (str): Color used for terminal nodes
    Returns:
        None
    """
    node_pos = nx.get_node_attributes(G, 'pos')
    node_list = list(nx.get_node_attributes(G, 'prize').keys())
    node_size = [50 * (size+10) for size in list(nx.get_node_attributes(G, 'prize').values())]
    node_color = [
        terminal_color
        if is_terminal else node_color
        for node, is_terminal in nx.get_node_attributes(G, 'terminal').items()
    ]

    f, ax = plt.subplots(figsize=(15, 15))

    # Draws graph nodes
    nx.draw_networkx_nodes(
        G,
        pos=node_pos,
        nodelist=node_list,
        node_size=node_size,
        node_color=node_color,
        ax=ax
    )
    nx.draw_networkx_labels(
        G,
        pos=nx.get_node_attributes(G, 'pos'),
        labels=nx.get_node_attributes(G, 'prize'),
        font_size=16,
        font_color='white',
        ax=ax
    )

    # Draws graph edges
    nx.draw_networkx_edges(
        G,
        pos=nx.get_node_attributes(G, 'pos'),
        edge_color='k',
        width=0.3,
        style='--',
        ax=ax
    )
    nx.draw_networkx_edge_labels(
        G,
        pos=nx.get_node_attributes(G, 'pos'),
        edge_labels=nx.get_edge_attributes(G, 'cost'),
        font_color='blue'
    )
