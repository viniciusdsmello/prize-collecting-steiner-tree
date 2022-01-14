import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'white'

def draw_steiner_graph(
        G: nx.Graph,
        plot_title: str = None,
        steiner_graph: nx.Graph = None,
        node_label: str = 'prize',
        node_color: str = 'blue',
        terminal_color: str = 'red'
    ):
    """
    Function that receives a PCSTP Graph and draws its nodes and edges

    Args:
        G (nx.Graph): Prize-Collecting Steiner Tree Instance.
        plot_title (str): Plot title.
        steiner_graph (nx.Graph): Steiner-Tree solution.
        node_label (str): Label used for nodes ('prize' or 'name').
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
    node_label = nx.get_node_attributes(G, 'prize') if node_label == 'prize' else {n: n for n in G.nodes}

    f, ax = plt.subplots(figsize=(15, 15))

    ax.set_title(plot_title, fontsize = 14)

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
        labels=node_label,
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

    if steiner_graph:
        nx.draw_networkx_edges(
            steiner_graph,
            pos=nx.get_node_attributes(G, 'pos'),
            edge_color='orange',
            width=2,
            style='-.',
            ax=ax
        )