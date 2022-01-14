import re
import typing
import networkx as nx
from pcstp.steinertree import SteinerTreeProblem


class SteinlibReader():
    def __init__(self, default_node_prize: int = 0, **kwargs):
        self.STP = SteinerTreeProblem()
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

                self.STP.graph.add_edge(v, w, distance=distance)

            elif line.startswith("Nodes"):
                nodes = re.findall(r'Nodes (\d+)$', line)
                self.STP.num_nodes = int(nodes[0])

            elif line.startswith("Edges"):
                edges = re.findall(r'Edges (\d+)$', line)
                self.STP.num_edges = int(edges[0])

            elif "END" in line:
                break

        # Set default node attributes
        nx.set_node_attributes(self.STP.graph, name='prize', values=self._default_node_prize)
        nx.set_node_attributes(self.STP.graph, name='terminal', values=False)

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
                self.STP.terminals.add(v_terminal)
            if line.startswith("TP "):
                entries = re.findall(r'(\d+)', line)
                terminal_vector = [entry for entry in entries if entry.isdecimal()]

                assert len(terminal_vector) == 2, "The line must to have two values"
                v_terminal, prize = terminal_vector

                v_terminal = int(v_terminal)
                prize = int(prize)

                terminals_prizes.update({v_terminal: prize})
                terminals_flags.update({v_terminal: True})

                self.STP.terminals.add(v_terminal)
            elif line.startswith("Terminals"):
                terminal = re.findall(r'Terminals (\d+)$', line)

                self.STP.num_terminals = int(terminal[0])

            elif "END" in line:
                break
        nx.set_node_attributes(self.STP.graph, terminals_prizes, "prize")
        nx.set_node_attributes(self.STP.graph, terminals_prizes, "terminal")


class DatReader():
    def __init__(self, default_node_prize: int = 0):
        self.STP = SteinerTreeProblem()
        self._default_node_prize = default_node_prize

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
                if "SECTION Comment" in line:
                    self._parser_section_comment(file)

                elif "SECTION Graph" in line:
                    self._parser_section_graph(file)

                elif "SECTION Terminals" in line:
                    self._parser_section_terminals(file)
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

                self.STP.graph.add_edge(v, w, distance=distance)

            elif line.startswith("Nodes"):
                nodes = re.findall(r'Nodes (\d+)$', line)
                self.STP.num_nodes = int(nodes[0])

            elif line.startswith("Edges"):
                edges = re.findall(r'Edges (\d+)$', line)
                self.STP.num_edges = int(edges[0])

            elif "END" in line:
                break

        # Set default node attributes
        nx.set_node_attributes(self.STP.graph, name='prize', values=self._default_node_prize)
        nx.set_node_attributes(self.STP.graph, name='terminal', values=False)

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
                self.STP.terminals.add(v_terminal)
            if line.startswith("TP "):
                entries = re.findall(r'(\d+)', line)
                terminal_vector = [entry for entry in entries if entry.isdecimal()]

                assert len(terminal_vector) == 2, "The line must to have two values"
                v_terminal, prize = terminal_vector

                v_terminal = int(v_terminal)
                prize = int(prize)

                terminals_prizes.update({v_terminal: prize})
                terminals_flags.update({v_terminal: True})

                self.STP.terminals.add(v_terminal)
            elif line.startswith("Terminals"):
                terminal = re.findall(r'Terminals (\d+)$', line)

                self.STP.num_terminals = int(terminal[0])

            elif "END" in line:
                break
        nx.set_node_attributes(self.STP.graph, terminals_prizes, "prize")
        nx.set_node_attributes(self.STP.graph, terminals_prizes, "terminal")
