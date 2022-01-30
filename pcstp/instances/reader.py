import glob
import os
import re
import typing
import networkx as nx
from pcstp.steinertree import SteinerTreeProblem


class SteinlibReader():
    def __init__(self, default_node_prize: int = 0, **kwargs):
        self.STP = SteinerTreeProblem()
        self.graph = nx.Graph()
        self.terminals = set()
        self._default_node_prize = default_node_prize

        self._num_nodes = 0
        self._num_edges = 0
        self._num_terminals = 0
        self._unique_nodes_parsed = set()

    def parser(self, filename: str):
        """
        Method responsible for parsing steinlib files and constructing a SteinerTreeProblem object
        Args:

            filename (str): Path to a .stp file
        Return:
            No return
        """
        self.STP.filename = filename

        # Open .stp files
        with open(filename, 'r') as file:
            for line in file:
                if "SECTION Comment" in line:
                    self._parser_section_comment(file)

                elif "SECTION Graph" in line:
                    self._parser_section_graph(file)

                elif "SECTION Terminals" in line:
                    self._parser_section_terminals(file)

        self.STP.terminals = self.terminals
        self.STP.graph = self.graph

        # assert self._num_nodes == len(self.graph.nodes), f'Failed to parse all nodes. Expected: {self._num_nodes} Parsed: {len(self.graph.nodes)}'
        # assert self._num_edges == len(self.graph.edges), f'Failed to parse all edges. Expected: {self._num_edges} Parsed: {len(self.graph.edges)}'
        # assert self._num_terminals == len(self.terminals), f'Failed to parse all terminals. Expected: {self._num_terminals} Parsed: {len(self.terminals)}'

        return self.STP

    def _parser_section_comment(self, file: typing.TextIO):
        """
        Method responsible for parsing the 'Comment' section of steinlib

        Args:
            file (typing.TextIO): File object being parsed
        Return:
            None
        """
        for line in file:
            _list = re.findall(r'"(.*?)"', line)
            if "Name" in line:
                _name = _list[0] if len(_list) else "Name unviable"
                self.STP.name = _name

            elif "Creator" in line:
                _creator = _list[0] if len(_list) else "Creator unviable"
                self.STP.creator = _creator

            elif "Remark" in line:
                remark = _list[0] if len(_list) else "Creator unviable"
                self.STP.remark = remark

            elif "END" in line:
                break

    def _parser_section_graph(self, file: typing.TextIO):
        """
        Method responsible for parsing the 'Graph' section of steinlib

        Args:
            file (typing.TextIO): File object being parsed
        Return:
            None
        """
        for line in file:
            if line.startswith('Nodes'):
                self._num_nodes = int(re.findall(r'([-+]*\d+\.\d+|[-+]*\d+)', line)[0])
            elif line.startswith('Edges'):
                self._num_edges = int(re.findall(r'([-+]*\d+\.\d+|[-+]*\d+)', line)[0])
            elif line.startswith("E ") or line.startswith("E\t"):
                entries = re.findall(r'([-+]*\d+\.\d+|[-+]*\d+)', line)
                edge_vector = [entry for entry in entries]

                assert len(edge_vector) == 3, "The line must to have three values"
                v, w, cost = edge_vector

                v = int(v)
                w = int(w)
                self._unique_nodes_parsed.add(v)
                self._unique_nodes_parsed.add(w)
                cost = float(cost)

                self.graph.add_edge(v, w, cost=cost)
            elif "END" in line:
                break
        # Set default node attributes
        nx.set_node_attributes(self.graph, name='prize', values=self._default_node_prize)
        nx.set_node_attributes(self.graph, name='terminal', values=False)

    def _parser_section_terminals(self, file: typing.TextIO):
        """
        Method responsible for parsing the 'Terminals' section of steinlib

        Args:
            file (typing.TextIO): File object being parsed
        Return:
            None
        """
        terminals_prizes = {}
        terminals_flags = {}

        for line in file:
            if line.startswith('Terminals'):
                self._num_terminals = int(re.findall(r'([-+]*\d+\.\d+|[-+]*\d+)', line)[0])
            elif line.startswith("T ") or line.startswith("T\t"):
                entries = re.findall(r'(\d{1,}((\.\d{1,})){0,})', line)
                terminal_vector = [entry[0] for entry in entries]
                assert len(terminal_vector) == 1, "The line must to have one value"
                v_terminal = terminal_vector

                v_terminal = int(v_terminal)

                terminals_prizes.update({v_terminal: prize})
                terminals_flags.update({v_terminal: True})

                self.terminals.add(v_terminal)

                self.terminals.add(v_terminal)
            elif line.startswith("TP"):
                entries = re.findall(r'(\d{1,}((\.\d{1,})){0,})', line)
                terminal_vector = [entry[0] for entry in entries]

                assert len(terminal_vector) == 2, "The line must to have two values"
                v_terminal, prize = terminal_vector

                v_terminal = int(v_terminal)
                prize = float(prize)

                if prize > 0:
                    terminals_prizes.update({v_terminal: prize})
                    terminals_flags.update({v_terminal: True})

                    self.terminals.add(v_terminal)
            elif line.startswith("RootP"):
                entries = re.findall(r'(\d{1,}((\.\d{1,})){0,})', line)
                terminal_vector = [entry[0] for entry in entries]

                assert len(terminal_vector) == 1, "The line must to have one value"

                v_terminal = terminal_vector[0]
                v_terminal = int(v_terminal)

                self.terminals.add(v_terminal)

            elif "END" in line:
                break
        nx.set_node_attributes(self.graph, terminals_prizes, "prize")
        nx.set_node_attributes(self.graph, terminals_prizes, "terminal")


class DatReader():
    def __init__(self, default_node_prize: int = 0):
        self.STP = SteinerTreeProblem()
        self.graph = nx.Graph()
        self.terminals = set()
        self._default_node_prize = default_node_prize

        self.has_reached_nodes_end = False

    def parser(self, filename: str):
        """
        Method responsible for parsing .dat files and constructing a SteinerTreeProblem object
        Args:
            filename (str): Path to a .dat file
        Return:
            No return
        """
        self.STP.filename = filename

        # Open .stp files
        with open(filename, 'r') as file:
            for line in file:
                if 'node' in line or '#name' in line:
                    self._parser_section_nodes(file)

                if 'link' in line or ('n1' in line and 'n2' in line) or 'length' in line or self.has_reached_nodes_end:
                    self._parser_section_links(file)

        self.STP.terminals = self.terminals
        self.STP.graph = self.graph

        return self.STP

    def _parser_section_nodes(self, file: typing.TextIO):
        """
        Method responsible for parsing the 'noe' section of .dat files

        Args:
            file (typing.TextIO): File object being parsed
        Return:
            None
        """
        for line in file:
            if line.startswith("#name") or line.startswith("#node"):
                continue
            elif "link" not in line:
                entries = re.findall(r'(\d{1,}((\.\d{1,})){0,})', line)
                vector = [entry[0] for entry in entries]

                if len(vector) == 4:
                    node, v, h, prize = vector

                    node = int(float(node))
                    v = int(float(v))
                    h = int(float(h))
                    prize = float(prize)

                    is_terminal = prize > 0

                    self.graph.add_node(node, prize=prize, terminal=is_terminal, pos=(v, h))

                    if node not in self.graph.nodes:
                        self.graph.add_node(node, terminal=is_terminal, prize=prize)
                    if is_terminal:
                        self.terminals.add(node)
                else:
                    self.has_reached_nodes_end = True
                    break
            else:
                self.has_reached_nodes_end = True
                break

    def _parser_section_links(self, file: typing.TextIO):
        """
        Method responsible for parsing the 'links' section of .dat files

        Args:
            file (typing.TextIO): File object being parsed
        Return:
            None
        """
        for line in file:
            if line.startswith("#name") or line.startswith("#link") or 'ring' in line:
                continue
            elif 'node' not in line and 'link' not in line:
                entries = re.findall(r'(\d{1,}((\.\d{1,})){0,})', line)
                vector = [entry[0] for entry in entries]

                assert len(vector) == 4, "The line must to have 4 values"
                edge, u, v, cost = vector

                edge = int(float(edge))
                u = int(float(u))
                v = int(float(v))
                cost = float(cost)

                self.graph.add_edge(u, v, cost=cost)
            else:
                break


if __name__ == "__main__":
    stp_reader = SteinlibReader()

    INSTANCES_PATH_PREFIX = './data/instances/benchmark/'

    # all_files = glob.glob(os.path.join(INSTANCES_PATH_PREFIX, '*'), recursive=False)
    all_files = []
    for root, dirs, files in os.walk(INSTANCES_PATH_PREFIX):
        for file in files:
            if not file.endswith('.xlsx'):
                all_files.append(os.path.join(root, file))

    print(f"Importing {len(all_files)} files")

    files = all_files
    for filename in files:
        try:
            if filename.endswith('.stp'):
                stp_reader = SteinlibReader()
            else:
                stp_reader = DatReader()

            stp = stp_reader.parser(filename=filename)

            if stp.num_nodes == 0 or stp.num_edges == 0 or stp.num_terminals == 0:
                raise ValueError(f'Failed to parse file {filename}')

            # print('')
            # print("Filename: ", stp.filename)
            # print("Nodes: ", stp.num_nodes)
            # print("Edges: ", stp.num_edges)
            # print("Terminals: ", stp.num_terminals)
        except Exception as e:
            print(f"Failed to parse: {filename} - Error: {e}")
