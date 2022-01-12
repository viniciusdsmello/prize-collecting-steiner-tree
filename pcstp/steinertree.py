import networkx as nx

class SteinerTreeProblem():
    def __init__(self):
        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.num_terminals: int = 0

        self.graph = nx.Graph()
        self.terminals = set()

        self.name: str = None
        self.remark: str = None
        self.creator: str = None
        self.filename: str = None