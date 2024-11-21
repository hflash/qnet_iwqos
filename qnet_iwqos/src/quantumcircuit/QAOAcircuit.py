# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2023/6/26 16:04
# @Author   : deviludier @ LINKE
# @File     : QAOAcircuit.py
# @Software : PyCharm
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt

from src.quantumcircuit.circuit import QuantumCircuit
from src.quantumcircuit.gate import *

"""
This class is 
"""


class QAOAcircuit:
    def __init__(self, num_qubits, graph: Graph):
        self.num_qubits = num_qubits
        self.graph = graph
        # self.constructed_circuit()

    def Hamiltonian(self):
        H = []

        # if graph is a weighted graph
        for edge in self.graph.edges:
            if not len(edge) == 3:
                break
            tmp_dict = {'Z' + str(edge[0]) + 'Z' + str(edge[1]) : edge[-1]}
            H.append(tmp_dict)

        # if graph is an unweighted graph
        for edge in self.graph.edges:
            if not len(edge) == 2:
                break
            tmp_dict = {'Z ' + str(edge[0]) + ' Z ' + str(edge[1]) : 1}
            H.append(tmp_dict)
        return H

    def constructed_circuit(self, param):
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.add_gate(H(i))

        k = 0
        for edge in self.graph.edges:
            i = edge[0]
            j = edge[1]
            qc.add_gate(CX(i, j))
            qc.add_gate(RZ(i, para=0))
            # k += 1
            qc.add_gate(CX(i, j))

        for i in range(self.num_qubits):
            qc.add_gate(RX(i, para=0))
            # k += 1

        return qc


if __name__ == "__main__":
    # exact statevector simulation
    G = nx.random_regular_graph(3, 10)
    nx.draw(G, with_labels=True)
    plt.show()
    print(G.edges)
    num_qubits = G.number_of_nodes()
    print(num_qubits)
    qaoa_qc = QAOAcircuit(num_qubits=num_qubits, graph=G)
    print(qaoa_qc.Hamiltonian())
