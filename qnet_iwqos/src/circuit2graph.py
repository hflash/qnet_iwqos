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
# @Time     : 2024/2/5 20:35
# @Author   : HFLASH @ LINKE
# @File     : circuit2graph.py
# @Software : PyCharm
import os
import random

import math
import numpy as np
from qiskit import QuantumCircuit as IBMquantumcircuit
import networkx as nx
# import matplotlib.pyplot as plt
from metisCut import metis_zmz
from src.quantumcircuit.circuit import QuantumCircuit


# 这个文件的内容：
#   readCircuitIBM 从路径读取量子线路到IBM QuantumCricuit格式，但是后面用不上了
#   circuit2Graph 读取量子线路中的两比特量子门，并将之转化为无向带权图
#   本来打算用图的betweeness做划分的，后来发现不是特别合适
#   circuitPartition 将图通过metis方法（metis_zmz方法，带权图划分）划分成k个子图，k = 线路量子比特数目/分布式节点量子数目，假设分布式量子设备的计算比特数目是相同的
#       返回remote_operations 远程操作的门id
#       circuit_dagtable 量子线路的dagtable
#       gate_list 量子线路的门列表，其中的每一个元素都是LINKEQ的quantumgate
#       subcircuit_communication 子线路通信的邻接数组
#       qubit_loc_dic 量子比特与子线路的对应关系字典
#       subcircuit_qubit_partitions subcircuit i 中有哪些qubit
def readCircuitIBM(path):
    circuit = IBMquantumcircuit.from_qasm_file(path)
    # print(circuit)
    return circuit


# def circuit2Graph(circuit: IBMquantumcircuit):
#     circuitGraph = nx.Graph()
#     circuitGraph.add_nodes_from([i for i in range(circuit.num_qubits)])
#     gates = circuit.data
#     for gate in gates:
#         # print(gate.operation.name, end=': ')
#         # print(gate.operation.num_qubits, end=' ')
#         if gate.operation.num_qubits == 2:
#             if (gate.qubits[0].index, gate.qubits[1].index) not in circuitGraph.edges:
#                 # print(gate.qubits[0].index, gate.qubits[1].index)
#                 circuitGraph.add_edge(gate.qubits[0].index, gate.qubits[1].index, weight=1)
#             else:
#                 nowweight = circuitGraph[gate.qubits[0].index][gate.qubits[1].index]['weight']
#                 circuitGraph.add_edge(gate.qubits[0].index, gate.qubits[1].index, weight=nowweight + 1)
#     # betweenness = nx.betweenness_centrality(circuitGraph, normalized=True, weight='weight')
#     # print(betweenness)
#     # nx.draw(circuitGraph, with_labels=True, font_weight='bold')
#     # plt.show()
#     #
#     # 检测 线路 cx 3 0和cx 0 3是否同为一条边
#     # weights = [circuitGraph[u][v]['weight'] for u, v in circuitGraph.edges()]
#     # for u, v in circuitGraph.edges():
#     #     weight = circuitGraph[u][v]['weight']
#     #     print(u, v, weight)
#     return circuitGraph
#
#     # 画出这张图 https://stackoverflow.com/questions/14943439/how-to-draw-multigraph-in-networkx-using-matplotlib-or-graphviz
#     # pos = nx.random_layout(circuitGraph)
#     # names = {name: name for name in circuitGraph.nodes}
#     # nx.draw_networkx_nodes(circuitGraph, pos, node_color='b', node_size=250, alpha=1)
#     # nx.draw_networkx_labels(circuitGraph, pos, names, font_size=12, font_color='w')
#     # ax = plt.gca()
#     # for e in circuitGraph.edges:
#     #     ax.annotate("",
#     #                 xy=pos[e[1]], xycoords='data',
#     #                 xytext=pos[e[0]], textcoords='data',
#     #                 arrowprops=dict(arrowstyle="->", color="0",
#     #                                 shrinkA=10, shrinkB=10,
#     #                                 patchA=None, patchB=None,
#     #                                 connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
#     #                                                                        ),
#     #                                 ),
#     #                 )
#     # plt.axis('off')
#     # plt.show()

def circuit2Graph(circuit: QuantumCircuit):
    circuitGraph = nx.Graph()
    circuitGraph.add_nodes_from([i for i in range(circuit.qubit_number)])
    circuit_dagtable = circuit.to_dagtable()
    gate_list = circuit.gate_list
    for gate in gate_list:
        qubits = gate.get_qubits()
        if len(qubits) == 2:
            if (qubits[0], qubits[1]) not in circuitGraph.edges:
                # print(gate.qubits[0].index, gate.qubits[1].index)
                circuitGraph.add_edge(qubits[0], qubits[1], weight=1)
            else:
                nowweight = circuitGraph[qubits[0]][qubits[1]]['weight']
                circuitGraph.add_edge(qubits[0], qubits[1], weight=nowweight + 1)
    return circuitGraph


def graphBetweenessCentrality(graph):
    pass


def circuitPartition(path, device_qubit_number, randomseed):
    # circuitIBM = readCircuitIBM(path)
    random.seed(randomseed)
    np.random.seed(randomseed)
    circuitLINKEQ = QuantumCircuit.from_QASM(path)
    circuit_dagtable = circuitLINKEQ.to_dagtable()
    gate_list = circuitLINKEQ.gate_list
    circuitGraph = circuit2Graph(circuitLINKEQ)
    k = math.ceil(circuitLINKEQ.qubit_number / device_qubit_number)
    # gate_list 与global_operations也就是dagtable对应，但是gatelist中有一个init门，在后两者中自动隐去
    # cuts 与 cut_weight_sum一致
    # membership: 节点属于哪个子图
    # cut_weight_sum: 被分割掉的边的数目
    cuts, membership, cut_weight_sum = metis_zmz(circuitGraph, k, randomseed)
    # print(cuts, membership, cut_weight_sum)
    # subcircuits = []
    remote_operations = []

    # qubit 与 子线路之间的关系， subcircuit_qubit_partitions[0]中存储子线路0上的所有qubit
    subcircuit_qubit_partitions = [[] for _ in range(k)]

    # qubit 到 partition的字典
    qubit_loc_dic = {}
    for index, value in enumerate(membership):
        qubit_loc_dic[index] = value
    for key in qubit_loc_dic.keys():
        subcircuit_qubit_partitions[qubit_loc_dic[key]].append(key)
    subcircuit_communication = np.zeros([k, k], dtype=int)
    # for cut in range(cuts):
    #     subcircuit_qubit = [index for index, value in enumerate(membership) if value == cut]
    #     subcircuit_qubit_partitions.append(subcircuit_qubit)
    notSameLocationNum = 0
    for num, gate in enumerate(gate_list):
        if num == 0:
            # 忽略第一个门
            assert gate.name == 'Init'
            continue
        qubits = gate.get_qubits()
        if len(qubits) > 1:
            # assert gate.name == "CX"
            same_location_flag = (qubit_loc_dic[qubits[0]] == qubit_loc_dic[qubits[1]])
            # print(same_location_flag)
            if not same_location_flag:
                subcircuit_communication[qubit_loc_dic[qubits[0]]][qubit_loc_dic[qubits[1]]] += 1
                subcircuit_communication[qubit_loc_dic[qubits[1]]][qubit_loc_dic[qubits[0]]] += 1
                notSameLocationNum += 1
                remote_operations.append(num)
    # 切割掉的边的权重之和应当与上面过程中属于不同子线路的CX门数目一致
    assert cut_weight_sum == notSameLocationNum and notSameLocationNum == len(remote_operations)
    # print(remote_operations)
    return remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions


if __name__ == "__main__":
    # dir = ''
    # path = './adder_n4.qasm'
    # circuit = readCircuit(path)
    # circuit2Graph(circuit)
    path = '../exp_circuit_benchmark/large_scale'
    # print(circuitPartition(path))
    small_data = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.qasm'):
                circuit_name = file.split('.')[0]
                qasm_path = os.path.join(root, file)
                print(qasm_path)
                with open(qasm_path, 'r') as f:
                    circuit = QuantumCircuit.from_QASM(qasm_path)
                    qubit_num = circuit.qubit_number
                    depth = circuit.get_circuit_depth()
                    two_qubit_gate_num = 0
                    gate_list = circuit.gate_list
                    for gate in gate_list:
                        qubits = gate.get_qubits()
                        if len(qubits) == 2:
                            two_qubit_gate_num += 1
                    path = os.path.join(root, file)
                    remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions = circuitPartition(
                        path, 40, randomseed=np.random.seed())
                    small_data[circuit_name] = [qubit_num, depth, two_qubit_gate_num, len(remote_operations)]
    print(small_data)

    # graph = nx.Graph()
    # graph.add_weighted_edges_from([(1, 2, 2), (1, 3, 1), (2, 3, 1), (3, 4, 1), (2, 4, 1)])
    # betweeness = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    # print(betweeness)
