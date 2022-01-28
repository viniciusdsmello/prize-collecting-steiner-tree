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
            if line.startswith("E "):
                entries = re.findall(r'(\d+)', line)
                edge_vector = [entry for entry in entries if entry.isdecimal()]

                assert len(edge_vector) == 3, "The line must to have three values"
                v, w, distance = edge_vector

                v = int(v)
                w = int(w)
                distance = int(distance)

                self.graph.add_edge(v, w, cost=distance)
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
            if line.startswith("T "):
                entries = re.findall(r'(\d+)', line)
                terminal_vector = [entry for entry in entries if entry.isdecimal()]
                assert len(terminal_vector) == 1, "The line must to have one value"
                v_terminal = terminal_vector

                v_terminal = int(v_terminal)

                terminals_prizes.update({v_terminal: prize})
                terminals_flags.update({v_terminal: True})

                self.terminals.add(v_terminal)

                self.terminals.add(v_terminal)
            if line.startswith("TP "):
                entries = re.findall(r'(\d+)', line)
                terminal_vector = [entry for entry in entries if entry.isdecimal()]

                assert len(terminal_vector) == 2, "The line must to have two values"
                v_terminal, prize = terminal_vector

                v_terminal = int(v_terminal)
                prize = int(prize)

                terminals_prizes.update({v_terminal: prize})
                terminals_flags.update({v_terminal: True})

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
                    prize = int(float(prize))

                    is_terminal = prize > 0

                    self.graph.add_node(node, prize=prize, terminal=is_terminal, pos=(v, h))

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
                cost = int(float(cost))

                self.graph.add_edge(u, v, cost=cost)
            else:
                break


if __name__ == "__main__":
    stp_reader = DatReader()
    # stp = stp_reader.parser(filename='./data/instances/dat/steinc1-wmax_10-seed_33000-gw.dat')
    stp = stp_reader.parser(filename='./data/instances/dat/K100')

    print("Nodes: ", stp.num_nodes)
    print("Edges: ", stp.num_edges)
    print("Terminals: ", stp.num_terminals)
