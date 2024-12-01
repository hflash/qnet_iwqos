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
# @Time    : 2024/2/12 21:46
# @Author  : HFALSH @ LINKE
# @File    : distributed_operation_circuit.py
# @IDE     : PyCharm
import os
import random
import time

from scipy.sparse.csgraph import dijkstra
# from circuit2graph import circuitPartition
import src.main_cd_circuit_execution as cd
import numpy as np
from src.matrix2matrix import map_nodes
from src.circuit2graph import circuitPartition
import heapq
from src.quantumcircuit.remotedag import RemoteDag


def circuit_partition_trial():
    random.seed(0)
    path = './adder_n4.qasm'
    remote_operations, operations_dagtable = circuitPartition(path)
    print(circuitPartition(path))


def cd_trial():
    ## PROTOCOL
    protocol = 'srs'  # Currently only 'srs' has been debugged

    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    # Here we use a squared lattice (with hard boundary conditions)
    # with 9 nodes as an example.
    l = 3
    n = int(l * l)
    A = cd.adjacency_squared_hard(l)
    topology = 'squared_hard'

    ## HARDWARE
    p_gen = 1  # Probability of successful entanglement generation
    p_swap = 1  # Probability of successful swap
    qbits_per_channel = 1  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = 0.12  # Probability of performing swaps in the SRS protocol
    max_links_swapped = 4  #  Maximum number of elementary links swapped
    p_cons = 0  # Probability of virtual neighbors consuming a link per time step
    F_app = 0  # Minimum fidelity required by the background application

    ## CUTOFF
    # The cutoff is here chosen arbitrarily. To find a physically meaningful value,
    # one should use the coherence time of the qubits and the fidelity of newly
    # generated entangled links. A valid approach is to use a worst-case model
    # as in Iñesta et al. 'Optimal entanglement distribution policies in homogeneous
    # repeater chains with cutoffs', 2023.
    cutoff = 20

    ## SIMULATION
    data_type = 'avg'  # Store only average (and std) values instead of all simulation data
    N_samples = 1  #  Number of samples
    total_time = int(cutoff * 5)  #  Simulation time
    randomseed = 0
    np.random.seed(randomseed)
    cd.simulation_cd(protocol, A, p_gen, q_swap, p_swap,
                     p_cons, cutoff, max_links_swapped,
                     qbits_per_channel, N_samples,
                     total_time,
                     return_data=data_type)


def trivial_allocate_subcircuit(subcircuits, subcircuit_communication, adjacency_matrix):
    """
    以trivial的方式得到子线路到量子网络拓扑结构的映射
    具体为：
    子线路1 → 量子计算机节点1
    子线路2 → 量子计算机节点2
    :param subcircuits: 子线路数目
    :param subcircuit_communication: 子线路之间需要实现的通信
    :param topology: 用邻接矩阵表示
    :return:
    """
    subcircuits_allocation = {}
    for index in range(subcircuits):
        subcircuits_allocation[index] = index
    # subcircuits_allocation[0] = 2
    # subcircuits_allocation[1] = 8
    return subcircuits_allocation


def Hungarian_allocate_subcircuit(subcircuits, subcircuit_communication, physical_adjacency_matrix):
    """
    以trivial的方式得到子线路到量子网络拓扑结构的映射
    具体为：
    子线路1 → 量子计算机节点1
    子线路2 → 量子计算机节点2
    :param subcircuits: 子线路数目
    :param subcircuit_communication: 子线路之间需要实现的通信
    :param physical_adjacency_matrix: 用邻接矩阵表示
    :return:
    """
    subcircuits_allocation = {}
    subcircuits, nodes = map_nodes(subcircuit_communication, physical_adjacency_matrix)
    for index, value in enumerate(subcircuits):
        subcircuits_allocation[value] = nodes[index]
    # subcircuits_allocation[0] = 2
    # subcircuits_allocation[1] = 8
    return subcircuits_allocation


def get_first_layer(current_dagtable):
    front_set = set()
    for i in range(len(current_dagtable)):
        if current_dagtable[i][0] >= 0:
            front_set.add(current_dagtable[i][0])
    gate_ids = list(front_set)
    return gate_ids


def get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list):
    """
    :param qubit_executed_gate: 每个qubit上最新执行的门列表qubit_executed_gate[i]对应qubit i最新执行的量子门
    :param circuit_dagtable: 线路的原始dagtable
    :param gate_list: 线路中所有门列表
    :param executed_gate_list: 已经执行的门列表
    :return:
    """
    front_set = set()
    # circuit_dagtable中取每个比特对应的门列表
    for qubit, qubit_gate_list in enumerate(circuit_dagtable):
        # 从比特对应的门列表中找到最新执行的比特的门
        for index, gate_id in enumerate(qubit_gate_list):
            # 匹配到相应门，且这个门不是这个比特上的最后一个门
            if qubit_executed_gate[qubit] == -2:
                for gate_id_qubit in circuit_dagtable[qubit]:
                    if gate_id_qubit != -1:
                        front_set.add(gate_id_qubit)
            if qubit_executed_gate[qubit] == gate_id and qubit_gate_list[index] != qubit_gate_list[-1]:
                # 从该比特的下一个门开始，找到不是-1的第一个门，将它加入front_layer
                for i in range(index + 1, len(qubit_gate_list)):
                    if qubit_gate_list[i] != -1:
                        front_set.add(qubit_gate_list[i])
                        break
    # 检查front_layer中的每一个门，看这个门是否是两比特门，若是，查看这个门的一个或两个前驱是否在executed_gate_list中，若不在，则从front_layer中删去
    front_layer = list(front_set)
    for gate_id_front in front_set:
        gate = gate_list[gate_id_front]
        qubits = gate.get_qubits()
        predecessors = []
        if len(qubits) == 2:
            # assert gate.name == "CX"
            for index, gate_id in enumerate(circuit_dagtable[qubits[0]]):
                if gate_id_front == gate_id:
                    # 从后往前找到这个门的直接前驱： 第一个比特
                    for i in range(index - 1, -1, -1):
                        if circuit_dagtable[qubits[0]][i] != -1:
                            predecessors.append(circuit_dagtable[qubits[0]][i])
                            break
                    # 从后往前找到这个门的直接前驱：第二个比特
                    for i in range(index - 1, -1, -1):
                        if circuit_dagtable[qubits[1]][i] != -1:
                            predecessors.append(circuit_dagtable[qubits[1]][i])
                            break
                    if len(predecessors) > 0:
                        break
        if len(qubits) == 1:
            for index, gate_id in enumerate(circuit_dagtable[qubits[0]]):
                if gate_id_front == gate_id:
                    # 从后往前找到这个门的直接前驱： 第一个比特
                    for i in range(index - 1, -1, -1):
                        if circuit_dagtable[qubits[0]][i] != -1:
                            predecessors.append(circuit_dagtable[qubits[0]][i])
                            break
        for predecessor in predecessors:
            if predecessor not in executed_gate_list:
                front_layer.remove(gate_id_front)
                break

    return front_layer


def show_entanglement_status(S):
    """
        初始化这样一个S矩阵，S[i][j][m]表示node i和node j之间的物理链路，m表示缓冲大小（用于该链路的比特数目）；整个过程持续维护这样一个矩阵
        % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
        % S[i][j][m][0]: age of the qubit (<cutoff+1).
        % S[i][j][m][1]: number of times its link was involved in swaps (<M).
        % S[i][j][m][2]: address of the qubit with which it is entangled.
        % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
        % S[i][j]=0.0, if nodes i and j do not share a physical link.
        % S[i][j][m]=None, if qubit (i,j,m) is unused.
        """
    n = len(S)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if S[i][j] == 0:
                continue
            for index, entanglement in enumerate(S[i][j]):
                if entanglement is None:
                    print("The qubit %d between node %d and node %d is idle." % (index, i, j))
                    continue
                print("The entanglement %d between node %d and node %d:" % (index, i, entanglement[2][0]))
                print("age: %d" % entanglement[0])
                print("number of swap times: %d" % (entanglement[1]))
                print("address of the qubit with which it is entangled:", entanglement[2])
                print("---------------------------------------------------------------------")


def execute_gate(gate_id, circuit_dagtable):
    # 每个门只能被删除一次（单比特门）或两次（两比特门）
    # 已经不再使用, 现在使用的是直接求dag table中front layer的方法
    execution = 0
    for i in range(len(circuit_dagtable)):
        for j in range(len(circuit_dagtable[i])):
            if circuit_dagtable[i][j] == gate_id:
                circuit_dagtable[i][j] = -2
                execution += 1
    assert execution == 1 or execution == 2
    return circuit_dagtable


def generate_entanglement_path_dijkstra(s, d, adjacency_matrix):
    dist_matrix, predecessors = dijkstra(csgraph=adjacency_matrix, directed=False, indices=[s], unweighted=True,
                                         return_predecessors=True)
    # shortest_path_length = dist_matrix[0, d]
    # print(f"Shortest path length from node {s} to node {d}: {shortest_path_length}")
    path = []
    current_node = d
    while current_node != s:
        if current_node == -9999:
            return None
        path.insert(0, current_node)
        current_node = predecessors[0, current_node]
    path.insert(0, s)
    # print(f"Shortest path from node {s} to node {d}: {path}")
    return path


def find_entanglement_paths_dfs(v_adj, s, d, max_length, path=[]):
    path = path + [s]
    if s == d and len(path) <= max_length + 1:
        return [path]
    if len(path) > max_length + 1:
        return []
    paths = []
    for node in range(len(v_adj[s])):
        if v_adj[s][node] != 0:  # Check if an edge exists
            if node not in path:  # Avoid cycles
                newpaths = find_entanglement_paths_dfs(v_adj, node, d, max_length, path)
                for newpath in newpaths:
                    paths.append(newpath)
    return paths


# 获取当前的虚拟拓扑
def virtual_adjacency_matrix(S):
    num_nodes = len(S)
    virtual_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    # virtual_link_age = np.zeros((num_nodes, num_nodes), dtype=int)
    # virtual_link_swap_num = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if S[i][j] != 0 and S[i][j] is not None:
                for entanglement in S[i][j]:
                    if entanglement is not None:
                        virtual_adj[i][entanglement[2][0]] += 1
                        # 遍历到j节点的时候还会再加一次，因此不需要下面这一行
                        # virtual_adj[entanglement[2][0]][i] += 1
    return virtual_adj


def check_entanglements(s, d, S):
    """

    :param s: source node
    :param d: destination node
    :param S: entanglement matrix
    :return:
    """
    """
            % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
            % S[i][j][m][0]: age of the qubit (<cutoff+1).
            % S[i][j][m][1]: number of times its link was involved in swaps (<M).
            % S[i][j][m][2]: address of the qubit with which it is entangled.
            % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
            % S[i][j]=0.0, if nodes i and j do not share a physical link.
            % S[i][j][m]=None, if qubit (i,j,m) is unused.
            """
    # 节点 8 与节点 2相连，但是S[2][8] = S[8][2] = 0, 这表示8 和 2没有物理链接，虚拟链接无法这样判定
    # 事实上，只能检索这两个node的每一个纠缠，看看是否与对方相连

    assert s != d
    check_list = [s, d]
    for check_node in check_list:
        for node, node_link in enumerate(S[check_node]):
            # 与邻居的纠缠
            if node_link != 0:
                for index, link in enumerate(node_link):
                    # 该link中比特正共享纠缠
                    if link is not None:
                        link_destination = link[2][0]
                        link_loc_node = link[2][1]
                        link_loc_qubit = link[2][2]
                        if link_destination == s and check_node == d:
                            return True
                        elif link_destination == d and check_node == s:
                            return True
    return False


def consuming_entanglements(s, d, S):
    """

    :param s: source node
    :param d: destination node
    :param S: entanglement matrix
    :return:
    """
    """
            % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
            % S[i][j][m][0]: age of the qubit (<cutoff+1).
            % S[i][j][m][1]: number of times its link was involved in swaps (<M).
            % S[i][j][m][2]: address of the qubit with which it is entangled.
            % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
            % S[i][j]=0.0, if nodes i and j do not share a physical link.
            % S[i][j][m]=None, if qubit (i,j,m) is unused.
            """
    # 节点 8 与节点 2相连，但是S[2][8] = S[8][2] = 0, 这表示8 和 2没有物理链接，虚拟链接无法这样判定
    # 事实上，只能检索这两个node的每一个纠缠，看看是否与对方相连
    assert s != d
    check_list = [s, d]
    for check_node in check_list:
        for node, node_link in enumerate(S[check_node]):
            # 与邻居的纠缠
            if node_link != 0:
                for index, link in enumerate(node_link):
                    # 该link中比特正共享纠缠
                    if link is not None:
                        link_destination = link[2][0]
                        link_loc_node = link[2][1]
                        link_loc_qubit = link[2][2]
                        if link_destination == s and check_node == d:
                            # print(S[check_node][node][index])
                            S[check_node][node][index] = None
                            # print(S[link_destination][link_loc_node][link_loc_qubit])
                            S[link_destination][link_loc_node][link_loc_qubit] = None
                            break
                        elif link_destination == d and check_node == s:
                            # print(S[check_node][node][index])
                            S[check_node][node][index] = None
                            # print(S[link_destination][link_loc_node][link_loc_qubit])
                            S[link_destination][link_loc_node][link_loc_qubit] = None
                            break
    return S


def consuming_entanglements_old(s, d, S):
    """

    :param s: source node
    :param d: destination node
    :param S: entanglement matrix
    :return:
    """
    """
            % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
            % S[i][j][m][0]: age of the qubit (<cutoff+1).
            % S[i][j][m][1]: number of times its link was involved in swaps (<M).
            % S[i][j][m][2]: address of the qubit with which it is entangled.
            % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
            % S[i][j]=0.0, if nodes i and j do not share a physical link.
            % S[i][j][m]=None, if qubit (i,j,m) is unused.
            """
    # 节点 8 与节点 2相连，但是S[2][8] = S[8][2] = 0, 这表示8 和 2没有物理链接，虚拟链接无法这样判定
    # 事实上，只能检索这两个node的每一个纠缠，看看是否与对方相连
    assert s != d
    check_list = [s, d]
    execute_list = []
    for check_node in check_list:
        for node, node_link in enumerate(S[check_node]):
            # 与邻居的纠缠
            if node_link != 0:
                for index, link in enumerate(node_link):
                    # 该link中比特正共享纠缠
                    if link is not None:
                        link_destination = link[2][0]
                        link_loc_node = link[2][1]
                        link_loc_qubit = link[2][2]
                        age = link[0]
                        if link_destination == s and check_node == d:
                            # print(S[check_node][node][index])
                            # S[check_node][node][index] = None
                            # # print(S[link_destination][link_loc_node][link_loc_qubit])
                            # S[link_destination][link_loc_node][link_loc_qubit] = None
                            execute_list.append(
                                [age, [check_node, node, index], [link_destination, link_loc_node, link_loc_qubit]])
                        elif link_destination == d and check_node == s:
                            # print(S[check_node][node][index])
                            # S[check_node][node][index] = None
                            # print(S[link_destination][link_loc_node][link_loc_qubit])
                            # S[link_destination][link_loc_node][link_loc_qubit] = None
                            execute_list.append(
                                [age, [check_node, node, index], [link_destination, link_loc_node, link_loc_qubit]])
    ages = [e[0] for e in execute_list]
    max_age = max(ages)
    # print(ages)
    for execute_operation in execute_list:
        if execute_operation[0] == max_age:
            S[execute_operation[1][0]][execute_operation[1][1]][execute_operation[1][2]] = None
            S[execute_operation[2][0]][execute_operation[2][1]][execute_operation[2][2]] = None
    return S


def execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, v_adj, max_cost):
    """
    :param qubit1_loc: 第一个比特所在节点
    :param qubit2_loc: 第二个比特所在节点
    :param S: 当前网络中纠缠状态信息
    :param v_adj: 从S中得到的此刻量子网络中的虚拟链路信息（用邻接矩阵表示）
    :return:
    """
    # 首先检索qubit1_loc和qubit2_loc两个node上是否有直接相连的纠缠，有则直接使用
    # baseline策略: 最短路径+最短纠缠路径+利用现有的纠缠
    # 其他策略: 等待纠缠生成？
    assert qubit1_loc != qubit2_loc
    execution_node_path = []
    shortest_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, v_adj)
    if shortest_path is None:
        # print("Current %d and %d cannot communicate by any intermediate nodes." % (qubit1_loc, qubit2_loc))
        return None, S
    for index, intermediate_node in enumerate(shortest_path[0: -1]):
        execution_node_path.append(shortest_path[index])
        execution_node_path.append(shortest_path[index + 1])
        S = consuming_entanglements(shortest_path[index], shortest_path[index + 1], S)
    # print("execute succeed! %d %d" % (qubit1_loc, qubit2_loc))
    # print("Path: ", execution_node_path)
    # print("Path length %d " % (len(shortest_path) - 1))
    return execution_node_path, S


def execute_remote_operation_random_path(qubit1_loc, qubit2_loc, S, v_adj, max_cost):
    """
        :param qubit1_loc: 第一个比特所在节点
        :param qubit2_loc: 第二个比特所在节点
        :param S: 当前网络中纠缠状态信息
        :param v_adj: 从S中得到的此刻量子网络中的虚拟链路信息（用邻接矩阵表示）
        :return:
        """
    assert qubit1_loc != qubit2_loc
    execution_node_path = []
    paths = find_entanglement_paths_dfs(v_adj, qubit1_loc, qubit2_loc, max_cost, path=[])
    if paths is None or len(paths) == 0:
        # print("Current %d and %d cannot communicate by any intermediate nodes." % (qubit1_loc, qubit2_loc))
        return None, S
    if len(paths) > 1:
        random.shuffle(paths)
    path = paths[0]
    for index, intermediate_node in enumerate(path[0: -1]):
        execution_node_path.append(path[index])
        execution_node_path.append(path[index + 1])
        S = consuming_entanglements(path[index], path[index + 1], S)
    # print("execute succeed! %d %d" % (qubit1_loc, qubit2_loc))
    # print("Random chosen path", execution_node_path)
    # print("Path length %d " % (len(path) - 1))
    return execution_node_path, S


def execute_remote_operation_shortest_path_old(qubit1_loc, qubit2_loc, S, v_adj, max_cost):
    """
    :param qubit1_loc: 第一个比特所在节点
    :param qubit2_loc: 第二个比特所在节点
    :param S: 当前网络中纠缠状态信息
    :param v_adj: 从S中得到的此刻量子网络中的虚拟链路信息（用邻接矩阵表示）
    :return:
    """
    # 首先检索qubit1_loc和qubit2_loc两个node上是否有直接相连的纠缠，有则直接使用
    # baseline策略: 最短路径+最短纠缠路径+利用现有的纠缠
    # 其他策略: 等待纠缠生成？
    assert qubit1_loc != qubit2_loc
    execution_node_path = []
    shortest_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, v_adj)
    if shortest_path is None:
        # print("Current %d and %d cannot communicate by any intermediate nodes." % (qubit1_loc, qubit2_loc))
        return None, S
    for index, intermediate_node in enumerate(shortest_path[0: -1]):
        execution_node_path.append(shortest_path[index])
        execution_node_path.append(shortest_path[index + 1])
        S = consuming_entanglements_old(shortest_path[index], shortest_path[index + 1], S)
    # print("execute succeed! %d %d" % (qubit1_loc, qubit2_loc))
    # print("Path: ", execution_node_path)
    # print("Path length %d " % (len(shortest_path) - 1))
    return execution_node_path, S


def execute_remote_operation_random_path_old(qubit1_loc, qubit2_loc, S, v_adj, max_cost):
    """
        :param qubit1_loc: 第一个比特所在节点
        :param qubit2_loc: 第二个比特所在节点
        :param S: 当前网络中纠缠状态信息
        :param v_adj: 从S中得到的此刻量子网络中的虚拟链路信息（用邻接矩阵表示）
        :return:
        """
    assert qubit1_loc != qubit2_loc
    execution_node_path = []
    paths = find_entanglement_paths_dfs(v_adj, qubit1_loc, qubit2_loc, max_cost, path=[])
    if paths is None or len(paths) == 0:
        # print("Current %d and %d cannot communicate by any intermediate nodes." % (qubit1_loc, qubit2_loc))
        return None, S
    if len(paths) > 1:
        random.shuffle(paths)
    path = paths[0]
    for index, intermediate_node in enumerate(path[0: -1]):
        execution_node_path.append(path[index])
        execution_node_path.append(path[index + 1])
        S = consuming_entanglements_old(path[index], path[index + 1], S)
    # print("execute succeed! %d %d" % (qubit1_loc, qubit2_loc))
    # print("Random chosen path", execution_node_path)
    # print("Path length %d " % (len(path) - 1))
    return execution_node_path, S


def parallel_execute_remote_operation(current_remote_operation_info, S, vadj, max_length):
    # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
    execute_results = [[] for _ in range(len(current_remote_operation_info))]
    execute_paths = {}
    parallel_execute_paths = {}
    gates = list(current_remote_operation_info.keys())
    for gate_id in gates:
        execute_result = []
        execute_result.append(gate_id)
        qubit1_loc = current_remote_operation_info[gate_id][0]
        qubit2_loc = current_remote_operation_info[gate_id][1]
        paths = find_entanglement_paths_dfs(vadj, qubit1_loc, qubit2_loc, max_length)
        execute_paths[gate_id] = paths
    for i, gate_id in enumerate(gates):
        for j in range(i + 1, len(gates)):
            print(i, j)
            print(gates[i], gates[j])

    return execute_results, S


# fzchen
def my_dijkstra(graph_matrix, source, target):
    """
    基于邻接矩阵实现 Dijkstra 算法 (单源 -> 单目标)。

    参数：
    - graph_matrix: 二维列表形式的邻接矩阵
    - source: 源点索引
    - target: 目标点索引

    返回：
    - 最短路径距离和路径 (距离, 路径)。若不可达，返回 (float('inf'), [])。
    """
    n = len(graph_matrix)  # 节点总数
    dist = [100000] * n  # 初始化所有节点距离为无穷大
    dist[source] = 0  # 源点距离为 0
    prev = [None] * n  # 用于路径回溯
    visited = [False] * n  # 标记节点是否已访问
    priority_queue = [(0, source)]  # 优先队列 (距离, 节点)

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        # 如果已访问，跳过
        if visited[current_node]:
            continue
        visited[current_node] = True

        # 如果找到目标点，回溯路径并返回,同时给出最小带宽
        min_bandwidth = 100000
        if current_node == target:
            path = []
            one_bandwidth_edge = []
            while current_node is not None:
                path.append(current_node)
                last_node = current_node
                current_node = prev[current_node]
                if current_node is not None:
                    tmp_bandwidth = graph_matrix[last_node][current_node]
                    if tmp_bandwidth < min_bandwidth:
                        min_bandwidth = tmp_bandwidth
                    if tmp_bandwidth == 1:
                        if last_node < current_node:
                            one_bandwidth_edge.append((last_node, current_node))
                        else:
                            one_bandwidth_edge.append((current_node, last_node))
            return current_dist, path[::-1], min_bandwidth, one_bandwidth_edge  # 路径反转

        # 遍历邻居节点
        for neighbor in range(n):
            weight = 100000
            if graph_matrix[current_node][neighbor] != 0:
                weight = 1
            if weight != 100000 and not visited[neighbor]:  # 有效边且未访问
                new_dist = current_dist + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    # 如果队列耗尽仍未找到目标点，说明不可达
    return 100000, [], 0, []


# fzchen

def rank(vadj, gate, max_dist, remotedag: RemoteDag, exe_impact_cnt, F_cnt, successors_total_cnt):
    h_direct_weight_alhpa = 1
    h_indirect_weight_belta = 1
    h_indirect_weight_gama = 1
    h_global_exe_weight_delta = 1
    h_global_sr_weight_tao = 1

    fidelity_lambda = 0.5

    W_max = np.max(vadj)
    q1_loc, q2_loc = gate[1]
    assert W_max > 0
    h_direct = h_direct_weight_alhpa * vadj[q1_loc][q2_loc] / W_max

    h_indirect = h_indirect_weight_belta * gate[4] / W_max - h_indirect_weight_gama * gate[2] / max_dist

    exe_impact = h_global_exe_weight_delta * exe_impact_cnt / F_cnt

    successor_rate = 0
    if successors_total_cnt > 0:
        successor_rate = h_global_sr_weight_tao * remotedag.gate_dict[
            gate[0]].get_successors_cnt() / successors_total_cnt
    h_global = exe_impact + successor_rate
    return (pow(fidelity_lambda, gate[2])) * (h_direct + h_indirect + h_global)


# fzchen
# gate: [gate_id, [qubit1_loc, qubit2_loc], dist, path, min_bandwidth,one_bondwidth_path, rank]
def execute_remote_operation_greedy_rank(current_remote_operation_info, S, vadj, max_length, remotedag: RemoteDag):
    # 排序
    current_remote_gates_list = [list(x) for x in list(current_remote_operation_info.items())]

    max_dist = 0
    total_successors_cnt = 0

    executable_remote_gates_list = []
    for gate in current_remote_gates_list:
        qubit1_loc = gate[1][0]
        qubit2_loc = gate[1][1]
        dist, path, min_bandwidth, one_bondwidth_path = my_dijkstra(vadj, qubit1_loc, qubit2_loc)
        if dist < 100000:
            if dist > max_dist:
                max_dist = dist
            executable_remote_gates_list.append(gate + [dist, path, min_bandwidth, one_bondwidth_path])

        total_successors_cnt += remotedag.gate_dict[gate[0]].get_successors_cnt()

    gate_n = len(executable_remote_gates_list)
    exe_impact_matrix = [0] * gate_n
    for i in range(gate_n):
        path1 = executable_remote_gates_list[i][3]
        for j in range(i + 1, gate_n):
            path2 = executable_remote_gates_list[j][3]
            if bool(set(path1) & set(path2)):
                exe_impact_matrix[i] += 1
                exe_impact_matrix[j] += 1

    for i in range(gate_n):
        executable_remote_gates_list[i].append(
            rank(vadj, executable_remote_gates_list[i], max_dist, remotedag, exe_impact_matrix[i], gate_n,
                 total_successors_cnt))
    executable_remote_gates_list.sort(key=lambda x: x[3], reverse=True)

    execute_results = {}

    for index in range(gate_n):
        gate = executable_remote_gates_list[index]
        path = gate[3]
        if len(path) == 0 or len(path) > max_length:
            continue
        for i in range(0, len(path) - 1):
            if check_entanglements(path[i], path[i + 1], S) == False:
                break
        else:
            tmp_path = []
            for i in range(len(path) - 1):
                S = consuming_entanglements(path[i], path[i + 1], S)
                tmp_path.append(path[i])
                tmp_path.append(path[i + 1])
            execute_results[gate[0]] = tmp_path

    return execute_results, S


def virtual_srs_info(srs_configurations, N_samples, total_time):
    protocol = 'srs'
    N_samples = N_samples
    total_time = total_time
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    np.random.seed(srs_configurations['randomseed'])

    vdegrees, vneighs, vneighborhoods, _ = cd.simulation_cd_for_virtual_neighbors(protocol, A, p_gen, q_swap, p_swap,
                                                                                  p_cons, cutoff, max_links_swapped,
                                                                                  qbits_per_channel, N_samples,
                                                                                  total_time,
                                                                                  srs_configurations['randomseed'],
                                                                                  progress_bar=None, return_data='all')
    print(vneighborhoods)


def srs_config_squared_hard(qubit_per_channel, q_swap, cutoff, randomseed):
    srs_configurations = {}

    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    # Here we use a squared lattice (with hard boundary conditions)
    # with 9 nodes as an example.
    l = 3
    n = int(l * l)
    A = cd.adjacency_squared_hard(l)
    topology = 'squared_hard'
    srs_configurations['adj'] = A

    ## HARDWARE
    p_gen = 1  # Probability of successful entanglement generation
    p_swap = 1  # Probability of successful swap
    qbits_per_channel = qubit_per_channel  # Number of qubits per node per physical neighbor
    srs_configurations['p_gen'] = p_gen
    srs_configurations['p_swap'] = p_swap
    srs_configurations['qubits'] = qbits_per_channel

    ## SOFTWARE
    # q_swap = 0.12  # Probability of performing swaps in the SRS protocol
    max_links_swapped = 4  #  Maximum number of elementary links swapped
    p_cons = 0  # Probability of virtual neighbors consuming a link per time step
    srs_configurations['q_swap'] = q_swap
    srs_configurations['max_swap'] = max_links_swapped
    srs_configurations['p_cons'] = p_cons

    ## CUTOFF
    # cutoff = 20
    srs_configurations['cutoff'] = cutoff
    if randomseed is not None:
        srs_configurations['randomseed'] = randomseed
    else:
        srs_configurations['randomseed'] = random.seed()

    return srs_configurations


def srs_config_tree(qubit_per_channel, q_swap, cutoff, randomseed):
    srs_configurations = {}

    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    # Here we use a squared lattice (with hard boundary conditions)
    # with 9 nodes as an example.
    l = 3
    # n = int(l * l)
    # A = cd.adjacency_squared_hard(l)
    A = cd.adjacency_tree(2, 4)
    topology = 'tree'
    srs_configurations['adj'] = A

    ## HARDWARE
    p_gen = 1  # Probability of successful entanglement generation
    p_swap = 1  # Probability of successful swap
    qbits_per_channel = qubit_per_channel  # Number of qubits per node per physical neighbor
    srs_configurations['p_gen'] = p_gen
    srs_configurations['p_swap'] = p_swap
    srs_configurations['qubits'] = qbits_per_channel

    ## SOFTWARE
    # q_swap = 0.12  # Probability of performing swaps in the SRS protocol
    max_links_swapped = 4  #  Maximum number of elementary links swapped
    p_cons = 0  # Probability of virtual neighbors consuming a link per time step
    srs_configurations['q_swap'] = q_swap
    srs_configurations['max_swap'] = max_links_swapped
    srs_configurations['p_cons'] = p_cons

    ## CUTOFF
    # cutoff = 20
    srs_configurations['cutoff'] = cutoff
    if randomseed is not None:
        srs_configurations['randomseed'] = randomseed
    else:
        srs_configurations['randomseed'] = random.seed()

    return srs_configurations


def srs_config_chain(qubit_per_channel, q_swap, cutoff, randomseed):
    srs_configurations = {}

    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    # Here we use a squared lattice (with hard boundary conditions)
    # with 9 nodes as an example.
    l = 3
    # n = int(l * l)
    # A = cd.adjacency_squared_hard(l)
    A = cd.adjacency_chain(9)
    topology = 'chain'
    srs_configurations['adj'] = A

    ## HARDWARE
    p_gen = 1  # Probability of successful entanglement generation
    p_swap = 1  # Probability of successful swap
    qbits_per_channel = qubit_per_channel  # Number of qubits per node per physical neighbor
    srs_configurations['p_gen'] = p_gen
    srs_configurations['p_swap'] = p_swap
    srs_configurations['qubits'] = qbits_per_channel

    ## SOFTWARE
    # q_swap = 0.12  # Probability of performing swaps in the SRS protocol
    max_links_swapped = 4  #  Maximum number of elementary links swapped
    p_cons = 0  # Probability of virtual neighbors consuming a link per time step
    srs_configurations['q_swap'] = q_swap
    srs_configurations['max_swap'] = max_links_swapped
    srs_configurations['p_cons'] = p_cons

    ## CUTOFF
    # cutoff = 20
    srs_configurations['cutoff'] = cutoff
    if randomseed is not None:
        srs_configurations['randomseed'] = randomseed
    else:
        srs_configurations['randomseed'] = random.seed()

    return srs_configurations


def trial_entanglement_consuming(srs_configurations):
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = 2  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    randomseed = 0
    np.random.seed(randomseed)

    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    print("Initial entanglement status:")
    show_entanglement_status(S)

    while True:
        S = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped)
        vadj = virtual_adjacency_matrix(S)
        if vadj[2][8] > 0:
            consuming_entanglements(2, 8, S)
            print("Initial entanglement status:")
            show_entanglement_status(S)
            break


def check_all_executed(executed_gate_list, gate_list):
    for index, gate in enumerate(gate_list):
        if index == 0:
            continue
        if index not in executed_gate_list:
            return False
    return True


def time_evolution_greedy(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
                          subcircuits_allocation,
                          remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :param schedule: 0 最短路径, 1 随机路径
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 生成远程门的dag
    node_cnt = len(virtual_adjacency_matrix(S))
    remotedag = RemoteDag(node_cnt, remote_operations, gate_list, qubit_loc_subcircuit_dic)

    # 确定当前直接可以执行的量子门
    current_gate = remotedag.get_front_layer()
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    # while not all_executed:
    while len(current_gate) > 0:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]
        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    remotedag.execute_gate(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj,
                                                                         max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path(qubit1_loc, qubit2_loc, S, vadj,
                                                                       max_links_swapped)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        remotedag.execute_gate(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        if schedule == 2:
            # for gate_id in current_gate:
            #     if gate_id not in remote_operations:
            #         executed_gate_list.append(gate_id)
            #         remotedag.execute_gate(gate_id)
            #         # execute_gate(gate_id, circuit_dagtable)
            #         gate = gate_list[gate_id]
            #         qubits = gate.get_qubits()
            #         for qubit in qubits:
            #             qubit_executed_gate[qubit] = gate_id
            #         this_step_operations.append([gate_id, 'local'])
            if len(current_remote_operation_info) == 1:
                # schedule 3 parallelize remote operations
                gate_id = list(current_remote_operation_info.keys())[0]
                gate = gate_list[gate_id]
                vadj = virtual_adjacency_matrix(S)
                qubit1_loc = current_remote_operation_info[gate_id][0]
                qubit2_loc = current_remote_operation_info[gate_id][1]
                path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj, max_links_swapped)
                if path is not None:
                    executed_gate_list.append(gate_id)
                    remotedag.execute_gate(gate_id)
                    ecost += len(path) / 2
                    for qubit in gate.get_qubits():
                        qubit_executed_gate[qubit] = gate_id
                    operation = [gate_id, path]
                    this_step_operations.append(operation)
                else:
                    operation = [gate_id, 'not executed']
                    this_step_operations.append(operation)
            elif len(current_remote_operation_info) > 1:
                # TODO
                # 写一个新的函数，替换下面的parallel_execute_remote_operation()
                # 应该也是一个一个执行（确定优先级、执行最高的那个门），执行完毕之后，回到这里，更新executed_gate_list
                # 更新front layer和current_remote_operation_info，循环直到当前所有的门都无法执行
                some_gate_exe_flag = True
                while some_gate_exe_flag:
                    vadj = virtual_adjacency_matrix(S)
                    # execute_results, S = parallel_execute_remote_operation(current_remote_operation_info, S, vadj,
                    #                                                        max_links_swapped)
                    execute_results, S = execute_remote_operation_greedy_rank(current_remote_operation_info, S, vadj,
                                                                              max_links_swapped, remotedag)
                    some_gate_exe_flag = False
                    for gate_id in execute_results.keys():
                        assert gate_id in remote_operations
                        path = execute_results[gate_id]
                        if len(path) > 0:
                            executed_gate_list.append(gate_id)
                            remotedag.execute_gate(gate_id)
                            ecost += len(path) / 2
                            gate = gate_list[gate_id]
                            qubits = gate.get_qubits()
                            for qubit in qubits:
                                qubit_executed_gate[qubit] = gate_id
                            operation = [gate_id, path]
                            this_step_operations.append(operation)
                            some_gate_exe_flag = True
                        else:
                            operation = [gate_id, 'not executed']
                            this_step_operations.append(operation)
                    # current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
                    current_gate = remotedag.get_front_layer()
                    current_remote_operation_info = {}
                    for gate_id in current_gate:
                        # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
                        # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
                        if gate_id in remote_operations:
                            gate = gate_list[gate_id]
                            qubit1 = gate.get_qubits()[0]
                            qubit2 = gate.get_qubits()[1]
                            qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                            qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                            current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        execution_schedule.append(this_step_operations)
        # current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        current_gate = remotedag.get_front_layer()
        S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
        time_step += 1
        discard_entanglement_count += counts
        # print("Current gates at time %d:" % time_step)
        # print(current_gate)
        # print("Entanglement status at time %d:" % time_step)
        # show_entanglement_status(S)
        # break
        #
        # all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count


def time_evolution(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic, subcircuits_allocation,
                   remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :param schedule: 0 最短路径, 1 随机路径
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 确定当前直接可以执行的量子门
    current_gate = get_first_layer(circuit_dagtable)
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    while not all_executed:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]
        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj,
                                                                         max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path(qubit1_loc, qubit2_loc, S, vadj,
                                                                       max_links_swapped)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        if schedule == 2:
            if len(current_remote_operation_info) == 1:
                # schedule 3 parallelize remote operations
                vadj = virtual_adjacency_matrix(S)
                path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj, max_links_swapped)
                if path is not None:
                    executed_gate_list.append(gate_id)
                    ecost += len(path) / 2
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    operation = [gate_id, path]
                    this_step_operations.append(operation)
                else:
                    operation = [gate_id, 'not executed']
                    this_step_operations.append(operation)
            elif len(current_remote_operation_info) > 1:
                vadj = virtual_adjacency_matrix(S)
                execute_results, S = parallel_execute_remote_operation(current_remote_operation_info, S, vadj,
                                                                       max_links_swapped)
                for execute_result in execute_results:
                    assert gate_id in remote_operations
                    gate_id = execute_result[0]
                    path = execute_result[1]
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        gate = gate_list[gate_id]
                        qubits = gate.get_qubits()
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        execution_schedule.append(this_step_operations)
        current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
        time_step += 1
        discard_entanglement_count += counts
        # print("Current gates at time %d:" % time_step)
        # print(current_gate)
        # print("Entanglement status at time %d:" % time_step)
        # show_entanglement_status(S)
        # break
        #
        all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count


def time_evolution_old(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
                       subcircuits_allocation,
                       remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 确定当前直接可以执行的量子门
    current_gate = get_first_layer(circuit_dagtable)
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    while not all_executed:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]

        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path_old(qubit1_loc, qubit2_loc, S, vadj,
                                                                             max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path_old(qubit1_loc, qubit2_loc, S, vadj, 5)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        if schedule == 2:
            if len(current_remote_operation_info) == 1:
                # schedule 3 parallelize remote operations
                vadj = virtual_adjacency_matrix(S)
                path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj, max_links_swapped)
                if path is not None:
                    executed_gate_list.append(gate_id)
                    ecost += len(path) / 2
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    operation = [gate_id, path]
                    this_step_operations.append(operation)
                else:
                    operation = [gate_id, 'not executed']
                    this_step_operations.append(operation)
            elif len(current_remote_operation_info) > 1:
                vadj = virtual_adjacency_matrix(S)
                execute_results, S = parallel_execute_remote_operation(current_remote_operation_info, S, vadj,
                                                                       max_links_swapped)
                for execute_result in execute_results:
                    assert gate_id in remote_operations
                    gate_id = execute_result[0]
                    path = execute_result[1]
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        gate = gate_list[gate_id]
                        qubits = gate.get_qubits()
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        execution_schedule.append(this_step_operations)
        current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
        time_step += 1
        discard_entanglement_count += counts
        # print("Current gates at time %d:" % time_step)
        # print(current_gate)
        # print("Entanglement status at time %d:" % time_step)
        # show_entanglement_status(S)
        # break
        #
        all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count


def time_evolution_only_remote_time(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
                                    subcircuits_allocation,
                                    remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 确定当前直接可以执行的量子门
    current_gate = get_first_layer(circuit_dagtable)
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    while not all_executed:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]

        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path_old(qubit1_loc, qubit2_loc, S, vadj,
                                                                             max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path_old(qubit1_loc, qubit2_loc, S, vadj, 5)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        if schedule == 2:
            if len(current_remote_operation_info) == 1:
                # schedule 3 parallelize remote operations
                vadj = virtual_adjacency_matrix(S)
                path, S = execute_remote_operation_shortest_path(qubit1_loc, qubit2_loc, S, vadj, max_links_swapped)
                if path is not None:
                    executed_gate_list.append(gate_id)
                    ecost += len(path) / 2
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    operation = [gate_id, path]
                    this_step_operations.append(operation)
                else:
                    operation = [gate_id, 'not executed']
                    this_step_operations.append(operation)
            elif len(current_remote_operation_info) > 1:
                vadj = virtual_adjacency_matrix(S)
                execute_results, S = parallel_execute_remote_operation(current_remote_operation_info, S, vadj,
                                                                       max_links_swapped)
                for execute_result in execute_results:
                    assert gate_id in remote_operations
                    gate_id = execute_result[0]
                    path = execute_result[1]
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        ecost += len(path) / 2
                        gate = gate_list[gate_id]
                        qubits = gate.get_qubits()
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        execution_schedule.append(this_step_operations)
        current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        if current_remote_operation_info != {}:
            S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
            time_step += 1
            discard_entanglement_count += counts
        all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count


def time_evolution_old_only_remote_time(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
                                        subcircuits_allocation,
                                        remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 确定当前直接可以执行的量子门
    #生成远程门的dag
    node_cnt = len(virtual_adjacency_matrix(S))
    remotedag = RemoteDag(node_cnt, remote_operations, gate_list, qubit_loc_subcircuit_dic)

    # 确定当前直接可以执行的量子门
    current_gate = remotedag.get_front_layer()
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    while len(current_gate) > 0:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]

        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path_old(qubit1_loc, qubit2_loc, S, vadj,
                                                                             max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path_old(qubit1_loc, qubit2_loc, S, vadj, 5)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        remotedag.execute_gate(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        execution_schedule.append(this_step_operations)
        print(this_step_operations)
        # current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        current_gate = remotedag.get_front_layer()
        if current_remote_operation_info != {}:
            S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
            time_step += 1
            discard_entanglement_count += counts
        # all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count

def time_evolution_old_only_remote_time_greedy(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
                                        subcircuits_allocation,
                                        remote_operations, schedule):
    """
    :param circuit_dagtable: 量子线路的dagtable, 用来配合get_front_layer获取当前最新需要执行的量子门
    :param gate_list: 量子线路门列表，用于从id到门信息的映射
    :param qubit_loc_dic: 量子比特到子线路的映射
    :param subcircuit_allocation: 子线路的映射，sub_circuit到设备id（node 1， node 2...）
    :param remote_operations: 远程操作列表
    :param srs_config: srs协议的相关参数
    :return:
    """
    # 返回值
    # 纠缠消耗ecost
    ecost = 0
    discard_entanglement_count = 0

    # TIME
    time_step = 0

    # 程序执行中判定
    all_executed = False  # all_executed: 判断整个程序是否执行完，通过判定executed_gate_list是不是等于线路中的门的数目来判定
    executed_gate_list = []  # executed_gate_list: 已经执行了的程序中的门
    qubit_executed_gate = [-2 for _ in range(len(circuit_dagtable))]
    # for gateslist in circuit_dagtable:
    #     print(gateslist)

    # 新的执行序列，包括远程操作的处理方案
    execution_schedule = []

    # TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    # HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    # SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    # CUTOFF
    cutoff = srs_configurations['cutoff']

    randomseed = srs_configurations['randomseed']
    random.seed(randomseed)
    np.random.seed(randomseed)
    # 初始化纠缠链接
    S = cd.create_qubit_registers(A, qbits_per_channel)
    # print("Initial entanglement status:")
    # show_entanglement_status(S)

    # 确定当前直接可以执行的量子门
    #生成远程门的dag
    node_cnt = len(virtual_adjacency_matrix(S))
    remotedag = RemoteDag(node_cnt, remote_operations, gate_list, qubit_loc_subcircuit_dic)

    # 确定当前直接可以执行的量子门
    current_gate = remotedag.get_front_layer()
    # print("Initial current gates:")
    # print(current_gate)

    # 除非线路执行完，否则循环不停止；设置一个时间界，超过该时间也强行停止
    while len(current_gate) > 0:
        # 本轮执行的门或执行方案
        this_step_operations = [time_step]

        # 初始化当前的远程量子操作, 通过gate_id获取门相应的量子比特，再将量子比特对应到子线路上，最后通过子线路的映射确定量子比特的具体node
        current_remote_operation_info = {}
        for gate_id in current_gate:
            # current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
            # 这个字典存储的是当前远程操作的信息，key为gate_id，值为相应的量子比特所在的量子节点
            if gate_id in remote_operations:
                gate = gate_list[gate_id]
                qubit1 = gate.get_qubits()[0]
                qubit2 = gate.get_qubits()[1]
                qubit1_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit1]]
                qubit2_loc = subcircuits_allocation[qubit_loc_subcircuit_dic[qubit2]]
                current_remote_operation_info[gate_id] = [qubit1_loc, qubit2_loc]
        if schedule == 0 or schedule == 1:
            # 首先执行本地量子操作
            for gate_id in current_gate:
                if gate_id not in remote_operations:
                    executed_gate_list.append(gate_id)
                    # execute_gate(gate_id, circuit_dagtable)
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    for qubit in qubits:
                        qubit_executed_gate[qubit] = gate_id
                    this_step_operations.append([gate_id, 'local'])
                else:
                    # 执行远程操作
                    assert gate_id in remote_operations
                    gate = gate_list[gate_id]
                    qubits = gate.get_qubits()
                    qubit1_loc = current_remote_operation_info[gate_id][0]
                    qubit2_loc = current_remote_operation_info[gate_id][1]
                    # print(gate_id, qubit1_loc, qubit2_loc)
                    # entanglement_path = generate_entanglement_path_dijkstra(qubit1_loc, qubit2_loc, A)
                    # print(entanglement_path)
                    vadj = virtual_adjacency_matrix(S)
                    path = None
                    # schedule 1 shortest path
                    if schedule == 0:
                        path, S = execute_remote_operation_shortest_path_old(qubit1_loc, qubit2_loc, S, vadj,
                                                                             max_links_swapped)
                    elif schedule == 1:
                        path, S = execute_remote_operation_random_path_old(qubit1_loc, qubit2_loc, S, vadj, 5)
                    if path is not None:
                        executed_gate_list.append(gate_id)
                        remotedag.execute_gate(gate_id)
                        ecost += len(path) / 2
                        for qubit in qubits:
                            qubit_executed_gate[qubit] = gate_id
                        operation = [gate_id, path]
                        this_step_operations.append(operation)
                    else:
                        operation = [gate_id, 'not executed']
                        this_step_operations.append(operation)
        execution_schedule.append(this_step_operations)
        print(this_step_operations)
        # current_gate = get_front_layer(qubit_executed_gate, circuit_dagtable, gate_list, executed_gate_list)
        current_gate = remotedag.get_front_layer()
        if current_remote_operation_info != {}:
            S, counts = cd.step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed)
            time_step += 1
            discard_entanglement_count += counts
        # all_executed = check_all_executed(executed_gate_list, gate_list)
    return ecost, time_step, execution_schedule, discard_entanglement_count


# print("Entanglement status at time %d:" % time_step)
# show_entanglement_status(S)
# break
#
def batch_circuit_execution(schedule, num_sample, small_device_qubit_number, large_device_qubit_number):
    path = '../exp_circuit_benchmark/small_scale'
    # print(circuitPartition(path))
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuitname = file.split('.qasm')[-2]
                print(qasm_path)
                device_qubit_number = 0
                # if "cm42" in qasm_path or "cm82" in qasm_path or "z4" in qasm_path:
                #     continue
                if 'small' in qasm_path:
                    scale = 'small'
                    device_qubit_number = small_device_qubit_number
                if 'large' in qasm_path:
                    scale = 'large'
                    device_qubit_number = large_device_qubit_number
                for i in range(num_sample):
                    path_write = '../exp_data/baseline_0221_unweighted/' + scale + '/' + circuitname + '_schedule_' + str(
                        schedule) + '_sample_' + str(
                        i) + "_trivial_allocation" + "_bandwidth_2" + '.qasm'
                    print(path_write)
                    randomseed = np.random.seed()
                    timestart = time.time()
                    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
                        qasm_path, device_qubit_number, randomseed)
                    srs_configurations = srs_config_squared_hard(randomseed)
                    subcircuits_allocation = trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                         subcircuits_communication,
                                                                         srs_configurations['adj'])
                    ecost, time_step, execution_schedule, discard_entanglement_count = time_evolution(
                        srs_configurations, circuit_dagtable,
                        gate_list, qubit_loc_subcircuit_dic,
                        subcircuits_allocation, remote_operations,
                        schedule=0)
                    process_time = time.time() - timestart
                    with open(path_write, 'w') as f:
                        f.write("qubit location dic: " + str(qubit_loc_subcircuit_dic))
                        f.write('\n')
                        f.write("subcircuit qubit partitions: " + str(subcircuit_qubit_partitions))
                        f.write('\n')
                        f.write("subcircuits allocation" + str(subcircuits_allocation))
                        f.write('\n')
                        f.write("Time cost: " + str(time_step))
                        f.write('\n')
                        f.write("Entanglement cost:" + str(ecost))
                        f.write('\n')
                        f.write("Processing time: " + str(process_time))
                        print("Write Succeed!")
                    # break
                    # print(qubit_loc_subcircuit_dic, subcircuit_qubit_partitions)
                    # print(subcircuits_allocation)
                    # print(ecost, time_step, execution_schedule)
            # break


def batch_srs_info():
    pass


if __name__ == "__main__":
    # schedules = [0, 1]
    # small_device_qubit_number = 5
    # large_device_qubit_number = 40
    # N_samples = 10
    # for schedule in schedules:
    #     batch_circuit_execution(schedule, N_samples, small_device_qubit_number, large_device_qubit_number)
    # qasm_path = '../exp_circuit_benchmark/small_scale/cm82a_208.qasm'
    import sys

    qasm_path = "/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/pra_benchmark/qft/qft_100.qasm"
    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
        qasm_path, device_qubit_number=40, randomseed=0)
    srs_configurations = srs_config_squared_hard(qubit_per_channel=1, q_swap=0.12, cutoff=10, randomseed=0)

    subcircuits_allocation = trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                         subcircuits_communication,
                                                         srs_configurations['adj'])
    # ecost, time_step, execution_schedule, discard_entanglement_count = time_evolution_greedy(srs_configurations,
    #                                                                                          circuit_dagtable,
    #                                                                                          gate_list,
    #                                                                                          qubit_loc_subcircuit_dic,
    #                                                                                          subcircuits_allocation,
    #                                                                                          remote_operations,
    #                                                                                          schedule=2)
    #
    ecost, time_step, execution_schedule, discard_entanglement_count = time_evolution_old_only_remote_time(srs_configurations,
                                                                                             circuit_dagtable,
                                                                                             gate_list,
                                                                                             qubit_loc_subcircuit_dic,
                                                                                             subcircuits_allocation,
                                                                                             remote_operations,
                                                                                             schedule=1)

    print(ecost, time_step, discard_entanglement_count)
    # cd_trial()
    # path = '../circuit_benchmark/qft/qft_newnew_100.qasm'
    # path = '../circuit_benchmark/qaoa/qaoa_max_cut_50.qasm'
    # path = '../circuit_benchmark/grover/grover_101.qasm'
    # path = './adder_n4.qasm'
    # path = '../circuit_benchmark/qaoa/qaoa_200.qasm'
    # path = '../circuit_benchmark/vqe/vqe_real_amplitudes_100_reps_full1.qasm'
    # path = '../circuit_benchmark/qaoa/qaoa_max_cut_50.qasm'
    # path = '../circuit_benchmark/rca/rca_100.qasm'
    # remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
    #     path, 2, 0)
    # srs_configurations = srs_config_squared_hard(0)
    # subcircuits_allocation = trivial_allocate_subcircuit(len(subcircuit_qubit_partitions), subcircuits_communication,
    #                                                      srs_configurations['adj'])
    # print(time_evolution(srs_configurations, circuit_dagtable, gate_list, qubit_loc_subcircuit_dic,
    #                      subcircuits_allocation, remote_operations, schedule=0))
    # choose_entanglement_path(2, 6, srs_configurations['adj'])
    # trial_entanglement_consuming(srs_configurations)
    # virtual_srs_info(srs_configurations, 100, 100)

# [[0.     0.9804 0.1225 0.9791 0.2669 0.0402 0.1136 0.0394 0.004 ]
#  [0.9804 0.     0.9782 0.3852 0.9906 0.3902 0.0421 0.0855 0.0391]
#  [0.1225 0.9782 0.     0.0391 0.2874 0.977  0.0041 0.0481 0.1041]
#  [0.9791 0.3852 0.0391 0.     0.9913 0.0878 0.9796 0.409  0.0314]
#  [0.2669 0.9906 0.2874 0.9913 0.     0.9922 0.2678 0.9925 0.2431]
#  [0.0402 0.3902 0.977  0.0878 0.9922 0.     0.0397 0.4109 0.9747]
#  [0.1136 0.0421 0.0041 0.9796 0.2678 0.0397 0.     0.9798 0.1033]
#  [0.0394 0.0855 0.0481 0.409  0.9925 0.4109 0.9798 0.     0.9711]
#  [0.004  0.0391 0.1041 0.0314 0.2431 0.9747 0.1033 0.9711 0.    ]]