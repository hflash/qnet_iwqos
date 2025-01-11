'''Main functions to analyze CD protocols.
   Alvaro G. Inesta. TU Delft, 2022.'''

import numpy as np
import numpy.matlib as npm
import json
import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import warnings
import scipy
from scipy import sparse
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn
import pandas as pd
from sys import getsizeof
import random
from pathlib import Path
import pickle
import os
import copy
from queue import Queue
import matplotlib.ticker as ticker
import networkx as nx
import heapq

plt.rcParams['font.family'] = 'Arial'


# rc('text', usetex=True)
# plt.rcParams.update({
#    'text.usetex': True,
#    'text.latex.preamble': r'\usepackage{amsfonts}'
# })

# ---------------------------------------------------------------------------
# ------------------------- PHYSICAL NETWORK --------------------------------
# ---------------------------------------------------------------------------
def adjacency_chain(n):
    '''Adjacency matrix of a repeater chain with n nodes
        (without periodic boundary conditions).'''
    A = np.zeros((n, n))
    A[0, 1] = 1
    A[-1, -2] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = 1
        A[i, i + 1] = 1
    return A


def adjacency_squared(l):
    '''Adjacency matrix of a squared grid with periodic boundary conditions.
        ---Inputs---
            · l:    (int) number of nodes per side of the grid.'''
    l = int(l)
    n = int(l ** 2)
    A = np.zeros((n, n))
    for i in range(n):
        # Top connections
        if i < l:
            A[i, (l ** 2 - 1) - (l - 1 - i)] = 1
        else:
            A[i, i - l] = 1
        # Bottom connections
        if i > (l - 1) * l - 1:
            A[i, i - (l ** 2 - 1) + l - 1] = 1
        else:
            A[i, i + l] = 1
        # Right connections
        if (i - (l - 1)) % l == 0:
            A[i, i - (l - 1)] = 1
        else:
            A[i, i + 1] = 1
        # Left connections
        if (i % l) == 0:
            A[i, i + (l - 1)] = 1
        else:
            A[i, i - 1] = 1
    return A


def adjacency_squared_hard(l):
    '''Adjacency matrix of a squared grid with hard boundary conditions.
        ---Inputs---
            · l:    (int) number of nodes per side of the grid.'''
    l = int(l)
    n = int(l ** 2)
    A = np.zeros((n, n))
    for i in range(n):
        # Top connections
        if i < l:
            pass
        else:
            A[i, i - l] = 1
        # Bottom connections
        if i > (l - 1) * l - 1:
            pass
        else:
            A[i, i + l] = 1
        # Right connections
        if (i - (l - 1)) % l == 0:
            pass
        else:
            A[i, i + 1] = 1
        # Left connections
        if (i % l) == 0:
            pass
        else:
            A[i, i - 1] = 1
    return A


def adjacency_dumbbell(n0, users1, users2):
    '''Adjacency matrix of a dumbbell network.
        ---Inputs---
            · n0:  (int) number of nodes in the main chain.
            · users1:   (list of ints) each element is the number
                        of users attached to a second-level repeater
                        in the first side of the network.
            · users2:   (list of ints) each element is the number
                        of users attached to a second-level repeater
                        in the second side of the network.'''
    n1 = len(users1)
    n2 = len(users2)
    n = n0 + n1 + n2 + sum(users1) + sum(users2)
    A = np.zeros((n, n))

    # Main chain: nodes 0 to n0-1
    for i in range(0, n0 - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1

    # Second-level repeaters - side 1
    for i in range(n0, n0 + n1):
        A[0, i] = 1
        A[i, 0] = 1

    # Second-level repeaters - side 2
    for i in range(n0 + n1, n0 + n1 + n2):
        A[n0 - 1, i] = 1
        A[i, n0 - 1] = 1

    # Users - side 1
    current_node = n0 + n1 + n2
    for idx, n1_ in enumerate(users1):
        for i in range(0, n1_):
            A[n0 + idx, current_node] = 1
            A[current_node, n0 + idx] = 1
            current_node += 1

    # Users - side 2
    for idx, n2_ in enumerate(users2):
        for i in range(0, n2_):
            A[n0 + n1 + idx, current_node] = 1
            A[current_node, n0 + n1 + idx] = 1
            current_node += 1
    return A


def adjacency_tree(d, k):
    '''Adjacency matrix of a (d,k)-tree-like network.
        ---Inputs---
            · d:    (int) number of child nodes per node.
            · k:    (int) number of levels.'''

    # Three level approach
    n = int((d ** k - 1) / (d - 1))
    A = np.zeros((n, n))
    lvl0_node = 0
    current_node = 1
    for node_lvl1 in range(d):
        A[lvl0_node, current_node] = 1
        A[current_node, lvl0_node] = 1
        lvl1_node = current_node
        current_node += 1
        for node_lvl2 in range(d):
            A[lvl1_node, current_node] = 1
            A[current_node, lvl1_node] = 1
            current_node += 1

    # Recurrent approach
    def new_level_loop(A, current_node, prevlvl_node, current_lvl):
        for node_currentlvl in range(d):
            A[prevlvl_node, current_node] = 1
            A[current_node, prevlvl_node] = 1
            current_node += 1
            if current_lvl < k - 1:
                A, current_node = new_level_loop(A, current_node, current_node - 1, current_lvl + 1)
        return A, current_node

    n = int((d ** k - 1) / (d - 1))
    A = np.zeros((n, n))
    A, _ = new_level_loop(A, 1, 0, 1)
    return A


def physical_degrees(A):
    '''Calculates the physical degree of each node.
        ---Inputs---
            · A:    (array) adjacency matrix.'''
    return np.sum(A, axis=0)


# ---------------------------------------------------------------------------
# ------------------------- BASIC OPERATIONS --------------------------------
# ---------------------------------------------------------------------------
def create_qubit_registers(A, r):
    '''Create a multidimensional array to store information about entangled
        states.
        ---Output---
            · S:    (array; int) S[i][j][m] contains the info about the qubit
                    with address (i,j,m), which is the qubit held by node i
                    in register (j,m), meaning that it was initially used to
                    generate an elementary link with node j.
                        S[i][j][m][0]: age of the qubit (<cutoff+1).
                        S[i][j][m][1]: number of times its link was involved
                                        in swaps (<M).
                        S[i][j][m][2]: address of the qubit with which it
                                        is entangled.
                    S[i][j]=0, if nodes i and j do not share a physical link.
                    S[i][j][m]=None, if qubit (i,j,m) is unused.'''
    n = len(A)
    if type(r) is int:
        S = np.zeros((n, n)).tolist()
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    S[i][j] = [None] * r
                    S[j][i] = [None] * r
    elif type(r) is list:
        assert len(r) == n
        S = np.zeros((n, n)).tolist()
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    assert r[i][j] == r[j][i]
                    S[i][j] = [None] * r[i][j]
                    S[j][i] = [None] * r[j][i]
    elif r == 'hetero_random':
        max_value = 5
        random.seed(3)
        S = np.zeros((n, n)).tolist()
        random_list = [random.randint(0, max_value) for _ in range(n)]
        physical_bandwidth = np.zeros((n, n)).tolist()
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    physical_bandwidth[i][j] = random_list[j]
                    physical_bandwidth[j][i] = random_list[j]
                    S[i][j] = [None] * random_list[j]
                    S[j][i] = [None] * random_list[j]
            random.shuffle(random_list)

    return S




def generate_all_links(S, p):
    '''Generates one entangled link per physical channel.'''
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            # If there is a physical channel i-j:
            if not S[i][j] == 0:
                for _ in range(len(S[i][j])):
                    # Find first available qubits
                    # generate_link_pair = [[None, None] for i in range(len(S[i][j]))]
                    qubit_number_i = None
                    for m_i, qubit_i in enumerate(S[i][j]):
                        if qubit_i == None:
                            qubit_number_i = m_i
                            # generate_link_pair[m_i][0] = m_i
                            break
                    qubit_number_j = None
                    for m_j, qubit_j in enumerate(S[j][i]):
                        if qubit_j == None:
                            qubit_number_j = m_j
                            # generate_link_pair[m_j][1] = m_j
                            break
                    # generate_link_pair = [[None, None] for i in range(len(S[i][j]))]
                    # qubit_number_i = None
                    # for m_i, qubit_i in enumerate(S[i][j]):
                    #     if qubit_i == None:
                    #         qubit_number_i = m_i
                    #         generate_link_pair[m_i][0] = m_i
                    #         # break
                    # qubit_number_j = None
                    # for m_j, qubit_j in enumerate(S[j][i]):
                    #     if qubit_j == None:
                    #         qubit_number_j = m_j
                    #         generate_link_pair[m_j][1] = m_j
                    #         # break
                    # ISSUE: could replace the previous loops by np.where, as
                    # in the following line. The problem is when S[i][j] has
                    # no 0 elements (i.e. all qubits i,j are occupied),
                    # which yields an error. To use the line below
                    # we must also change S[i][j] = [None]*r by
                    # S[i][j] = [0]*r in create_qubit_registers().
                    # qubit_number_i = np.where(np.asarray(S[i][j])==0)[0][0]

                    # If there are qubits available in nodes i and j, and
                    # with probability p:
                    if ((not qubit_number_i == None) and (not qubit_number_j == None)
                            and (np.random.rand() < p)):
                        # print(qubit_number_i, qubit_number_j)
                        # assert qubit_number_i==qubit_number_j
                        S[i][j][qubit_number_i] = [0, 0, [j, i, qubit_number_j]]
                        S[j][i][qubit_number_j] = [0, 0, [i, j, qubit_number_i]]
    return S


def advance_time(S):
    n = len(S)
    for i in range(n):
        for j in range(n):
            # If qubit register i-j exists:
            if not S[i][j] == 0:
                for m, qubit in enumerate(S[i][j]):
                    # If qubit is occupied:
                    if not qubit is None:
                        S[i][j][m][0] += 1
    return S


def cutoffs(S, cutoff):
    '''Resets qubits older than cutoff'''
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(n):
            # If qubit register i-j exists:
            if not S[i][j] == 0:
                for m, qubit in enumerate(S[i][j]):
                    # If qubit is occupied:
                    if not qubit is None:
                        age = qubit[0]
                        # If too old, reset qubit and entangled neighbor qubit:
                        if age >= cutoff:
                            S[qubit[2][0]][qubit[2][1]][qubit[2][2]] = None
                            S[i][j][m] = None
                            count += 1

    return S, count


def swap(S, qubit_id1, qubit_id2, ps, randomseed):
    '''Performs a probabilistic swap using the input qubits. Failed swaps
        produce a link with infinite age.
        ---Inputs---
            · S:    (array) qubit registers matrix
                    (see create_qubit_registers()).
            · qubit_id1:    (3-tuple) address of first qubit:
                            node holding qubit 1;
                            physical link towards which qubit 1 is oriented;
                            and index of qubit 1 in the physical link.
            · qubit_id2:    (3-tuple) address of second qubit:
                            node holding qubit 2;
                            physical link towards which qubit 2 is oriented;
                            and index of qubit 2 in the physical link.
            · ps:   (float) probability of successful swap.'''
    # random.seed(randomseed)
    # np.random.seed(randomseed)
    i1, j1, m1 = qubit_id1
    i2, j2, m2 = qubit_id2
    assert S[i1][j1], 'No qubit register [i1, j1]'
    assert not S[i1][j1][m1] == None, 'Qubit 1 is empty'
    assert S[i2][j2], 'No qubit register [i2, j2]'
    assert not S[i2][j2][m2] == None, 'Qubit 2 is empty'
    assert i1 == i2, 'Qubits belong to different nodes'
    # assert not j1==j2, 'Qubits belong to same physical channel'

    i1b, j1b, m1b = S[i1][j1][m1][2]
    i2b, j2b, m2b = S[i2][j2][m2][2]

    if np.random.rand() < ps:  # Successful swap
        new_age = max(S[i1][j1][m1][0], S[i2][j2][m2][0])
    else:  # Failed swap
        new_age = np.inf
    new_swaps = 1 + S[i1][j1][m1][1] + S[i2][j2][m2][1]

    S[i1b][j1b][m1b] = [new_age, new_swaps, [i2b, j2b, m2b]]
    S[i2b][j2b][m2b] = [new_age, new_swaps, [i1b, j1b, m1b]]

    S[i1][j1][m1] = None
    S[i2][j2][m2] = None

    # print("SWAP succeed!")
    return S


def remove_long_links(S, M):
    '''Resets qubits with links longer than M segments.'''
    n = len(S)
    count = 0
    for i in range(n):
        for j in range(n):
            # If qubit address (i,j,-) exists:
            if not S[i][j] == 0:
                for m, qubit in enumerate(S[i][j]):
                    # If qubit is occupied:
                    if not qubit is None:
                        link_length = qubit[1] + 1
                        # If too old, reset qubit and entangled neighbor qubit:
                        if link_length > M:
                            S[qubit[2][0]][qubit[2][1]][qubit[2][2]] = None
                            S[i][j][m] = None
                            count += 1
    return S, count


def consume_fixed_rate(S, cons_rate):
    '''Consumes links between each pair of virtual neighbors at rate
        cons_rate: floor(cons_rate) links are consumed deterministically
        and another link is consumed with probability cons_rate-floor(cons_rate).
        If the virtual neighbors do not share enough entangled links,
        they consume as many as possible.'''
    nodepairs_consumed = set()  # Pairs of nodes that already consumed links
    n = len(S)
    for node1 in range(n):
        _, _, vneighborhood1, vneigh_links = virtual_properties(S, node1)
        for node2 in range(node1, n):
            # Consume only if the nodes are virtual neighbors
            if ((node2 in vneighborhood1) and
                    (not {node1, node2} in nodepairs_consumed)):
                number_links_shared = len(vneigh_links[node2])
                cons_order = np.random.permutation(number_links_shared)
                k = -1
                # Consume deterministically only if cons_rate>=1
                for k in range(int(np.floor(cons_rate))):
                    cons_label = cons_order[k]
                    cons_qubit1 = vneigh_links[node2][cons_label]
                    cons_qubit2 = S[node1][cons_qubit1[1]][cons_qubit1[2]][2]
                    # Consume
                    S[node1][cons_qubit1[1]][cons_qubit1[2]] = None
                    S[node2][cons_qubit2[1]][cons_qubit2[2]] = None
                    if k > len(cons_order) - 1:
                        break
                # Consume probabilistically if cons_rate < 1 or if
                # cons_rate is not an integer
                rand = np.random.rand()
                # print(rand)
                if rand < cons_rate - np.floor(cons_rate):
                    k += 1
                    if k < len(cons_order):
                        cons_label = cons_order[k]
                        cons_qubit1 = vneigh_links[node2][cons_label]
                        cons_qubit2 = S[node1][cons_qubit1[1]][cons_qubit1[2]][2]
                        # Consume
                        # print(cons_qubit1, cons_qubit2)
                        S[node1][cons_qubit1[1]][cons_qubit1[2]] = None
                        S[node2][cons_qubit2[1]][cons_qubit2[2]] = None

                nodepairs_consumed.add(frozenset({node1, node2}))
    return S


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


def calculate_algebraic_connectivity(S):
    v_adj = virtual_adjacency_matrix(S)
    v_adj_1 = np.zeros((len(v_adj), len(v_adj)), dtype=int)
    for i in range(len(v_adj)):
        for j in range(len(v_adj[i])):
            if v_adj[i][j] == 0:
                continue
            else:
                v_adj_1[i][j] = 1
    degree_matrix = np.diag(np.sum(v_adj_1, axis=1))
    # 计算拉普拉斯矩阵
    laplacian_matrix = degree_matrix - v_adj_1
    # 计算特征值，返回第二小的特征值
    from scipy.linalg import eigh
    eigenvalues = eigh(laplacian_matrix, eigvals_only=True)
    original_connectivity = eigenvalues[1]
    assert original_connectivity + 0.000000000001 >= 0
    return original_connectivity


def calculate_algebraic_connectivity_benefit(S, pair, ps, randomseed=0):
    # original_connectivity = calculate_algebraic_connectivity(S)
    # import copy
    # new_S = copy.deepcopy(S)
    # node_i = pair[0]
    # node_j = pair[1]
    # new_S_after_swap = swap(new_S, node_i, node_j, randomseed)
    # assert S != new_S_after_swap
    # new_v_adj = virtual_adjacency_matrix(new_S_after_swap)
    # new_connectivity = calculate_algebraic_connectivity(new_v_adj)
    # return new_connectivity - original_connectivity
    original_connectivity = calculate_algebraic_connectivity(S)
    import copy
    new_S = copy.deepcopy(S)
    node_i = pair[0]
    node_j = pair[1]
    new_S_after_swap = swap(new_S, node_i, node_j, ps, randomseed)
    assert S != new_S_after_swap
    # new_v_adj = virtual_adjacency_matrix(new_S_after_swap)
    new_connectivity = calculate_algebraic_connectivity(new_S_after_swap)
    benefit = new_connectivity - original_connectivity
    # print("SWAP qubits:")
    # print(pair)
    # print(original_connectivity, new_connectivity, benefit)
    return benefit


def calculate_node_virtual_distances(S):
    v_adj = virtual_adjacency_matrix(S)
    # print(v_adj)
    n = len(v_adj)

    # 初始化距离矩阵
    # 使用高维矩阵，将非相连的边初始化为1000
    distance_matrix = np.full((n, n), 1000)

    # 将邻接矩阵转为距离矩阵
    for i in range(n):
        for j in range(n):
            if v_adj[i][j] >= 1:  # 如果存在边
                distance_matrix[i][j] = 1  # 直接相连的节点距离为 1
            if i == j:  # 自己到自己的距离为 0
                distance_matrix[i][j] = 0

    # 使用 Floyd-Warshall 算法更新距离矩阵
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = min(distance_matrix[i][j],
                                            distance_matrix[i][k] + distance_matrix[k][j])

    total_distances = 0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] is not np.inf:
                total_distances += distance_matrix[i][j]
    # print(distance_matrix)
    return total_distances


def calculate_node_virtual_distance_benefit(S, pair, ps, randomseed):
    original_distances = calculate_node_virtual_distances(S)
    import copy
    new_S = copy.deepcopy(S)
    node_i = pair[0]
    node_j = pair[1]
    new_S_after_swap = swap(new_S, node_i, node_j, ps, randomseed)
    assert S != new_S_after_swap
    # new_v_adj = virtual_adjacency_matrix(new_S_after_swap)
    new_distances = calculate_node_virtual_distances(new_S_after_swap)
    benifit = original_distances - new_distances
    # print(original_distances, new_distances, benifit)
    return benifit


def select_pairs(values, pairs):
    # 创建优先队列，并将每个 (value, pair) 插入优先队列
    max_heap = [(-value, pair) for value, pair in zip(values, pairs)]
    heapq.heapify(max_heap)  # 转换为优先队列

    used_elements = set()  # 用于标记已使用的 a 和 b
    selected_pairs = []  # 存放被选中的 pairs

    # 遍历优先队列
    while max_heap:
        neg_value, pair = heapq.heappop(max_heap)  # 取出最大值的 pair
        a, b = pair

        # 检查 a 和 b 是否已被使用
        if a not in used_elements and b not in used_elements:
            selected_pairs.append(pair)  # 选择这个 pair
            used_elements.add(a)  # 标记 a 为已使用
            used_elements.add(b)  # 标记 b 为已使用
    # print(selected_pairs)
    return selected_pairs


# ---------------------------------------------------------------------------
# ----------------------------- PROTOCOLS -----------------------------------
# ---------------------------------------------------------------------------
def step_protocol_cfs_connectivity_first_swap(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed,
                                              swap_mode):
    '''cfs protocol: connectivity first swap
    swap_mode: total_distance; algebraic_connectivity
    total_distance for executing entanglement swapping with the swap that can decrease total distances of across all nodes
    algebraic_connectivity for executing entanglement swapping with the swap that can increase algebraic connectivity of virtual adjacency matrix
    '''

    n = len(S)

    S = advance_time(S)
    discard_count = 0

    S, count = cutoffs(S, cutoff)
    discard_count += count

    # Generate links on every available qubit (1 per physical channel)
    S = generate_all_links(S, p_gen)
    # print("After generate:")
    # print(virtual_adjacency_matrix(S))

    # random.seed(randomseed)
    # np.random.seed(randomseed)

    # Perform swaps
    if q_swap > 0:
        node_list = np.random.permutation(np.arange(n))  # This does not need
        # to be a random permutation

        # Each node chooses a random pair
        pairs_list = [[] for _ in range(n)]
        for i in node_list:
            # Find all occupied qubits in node i
            # pairs_list_current: 所有可能的swap pair
            pairs_list_current = []
            # pair_list_good: 有正收益的swap pair
            pair_list_good = []
            value_list_good = []
            occupied_qubits_i = []
            for z in range(n):
                # If qubit address (i,z,-) exists
                if not S[i][z] == 0:
                    for m, qubit in enumerate(S[i][z]):
                        # If qubit is occupied:
                        if not qubit is None:
                            occupied_qubits_i += [(z, m)]
            if len(occupied_qubits_i) > 1:
                # Pick occupied qubit connected to node j
                for qubit_i in range(len(occupied_qubits_i)):
                    for qubit_j in range(qubit_i, len(occupied_qubits_i)):
                        if qubit_i != qubit_j:
                            pairs_list_current.append([occupied_qubits_i[qubit_i], occupied_qubits_i[qubit_j]])
                # random.shuffle(occupied_qubits_i)
                # qubit_1 = occupied_qubits_i[0]
                # j = S[i][qubit_1[0]][qubit_1[1]][2][0]
                # Pick occupied qubit connected to node k!=j, with A_jk=0
                # for qubit_2 in occupied_qubits_i:
                #     k = S[i][qubit_2[0]][qubit_2[1]][2][0]
                #     if ((not k == j) and
                #             (S[j][k] == 0)):  # A_jk=0 is equivalent to S_jk=0
                #         pairs_list[i] = [qubit_1, qubit_2]
                #         break
            benifits = []
            if swap_mode == 'total_distance':
                for pair in pairs_list_current:
                    swap_pair = [[i, pair[0][0], pair[0][1]],
                                 [i, pair[1][0], pair[1][1]]]
                    benifits.append(calculate_node_virtual_distance_benefit(S, swap_pair, ps = p_swap, randomseed=0))
            elif swap_mode == 'algebraic_connectivity':
                for pair in pairs_list_current:
                    swap_pair = [[i, pair[0][0], pair[0][1]],
                                 [i, pair[1][0], pair[1][1]]]
                    benifits.append(calculate_algebraic_connectivity_benefit(S, swap_pair, ps = p_swap, randomseed=0))
                    # print()
            else:
                print("unsupported swap mode")
            for m, value in enumerate(benifits):
                assert len(benifits) == len(pairs_list_current)
                if value > 0:
                    value_list_good.append(value)
                    pair_list_good.append(pairs_list_current[m])
                    # print(pairs_list_current[m])
            if len(pair_list_good) >= 1:
                pairs_list[i] = select_pairs(benifits, pairs_list_current)
            # print(virtual_adjacency_matrix(S))
            for pair in pairs_list[i]:
                swap_pair = [[i, pair[0][0], pair[0][1]],
                             [i, pair[1][0], pair[1][1]]]
                if (swap_mode == "total_distance" and calculate_node_virtual_distance_benefit(S, swap_pair, p_swap,
                                                                                              randomseed=randomseed) > 0) or (
                        swap_mode == "algebraic_connectivity" and calculate_algebraic_connectivity_benefit(S, swap_pair, p_swap,
                                                                                                           randomseed=randomseed) > 0):
                    S = swap(S, swap_pair[0],
                             swap_pair[1],
                             ps=p_swap, randomseed=randomseed)
                    # print("SWAP pair:")
                    # print(swap_pair)
                else:
                    continue
        # Each node attempts the swap
        # for i in node_list:
        #     # Perform a single swap
        #     if not pairs_list[i] is None:
        #         if np.random.rand() < q_swap:
        #             S = swap(S, [i, pairs_list[i][0][0], pairs_list[i][0][1]],
        #                      [i, pairs_list[i][1][0], pairs_list[i][1][1]],
        #                      ps=p_swap, randomseed=randomseed)
        # print("After SWAP:")
        # print(virtual_adjacency_matrix(S))
        # for i in node_list:
        #     if not pairs_list[i] is None:
        #         for pair in pairs_list[i]:
        #             print(pair)
        # S = swap(S, [i, pair[0][0], pair[0][1]],
        #                      [i, pair[1][0], pair[1][1]],
        #                      ps=p_swap, randomseed=randomseed)
        # print(virtual_adjacency_matrix(S))
    if p_swap < 1:
        S, count = cutoffs(S, cutoff + 10)  # We need to remove links from failed swaps,
        # which are links with infinite age. We need
        # to give them this age as a placeholder so
        # that all nodes can perform swaps at the
        # same time although in our simulation they
        # do this sequentally
        discard_count += count
    # Remove links that are too long
    # if q_swap > 0:
    S, count = remove_long_links(S, max_links_swapped)
    discard_count += count
    # Consume links
    S = consume_fixed_rate(S, p_cons)

    return S, discard_count


def step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, randomseed):
    '''SRS protocol: Single Random Swap'''
    n = len(S)

    S = advance_time(S)
    discard_count = 0

    S, count = cutoffs(S, cutoff)
    discard_count += count

    # Generate links on every available qubit (1 per physical channel)
    S = generate_all_links(S, p_gen)

    # random.seed(randomseed)
    # np.random.seed(randomseed)

    # Perform swaps
    if q_swap > 0:
        node_list = np.random.permutation(np.arange(n))  # This does not need
        # to be a random permutation

        # Each node chooses a random pair
        pairs_list = [None for _ in range(n)]
        for i in node_list:
            # Find all occupied qubits in node i
            occupied_qubits_i = []
            for z in range(n):
                # If qubit address (i,z,-) exists
                if not S[i][z] == 0:
                    for m, qubit in enumerate(S[i][z]):
                        # If qubit is occupied:
                        if not qubit is None:
                            occupied_qubits_i += [(z, m)]
            if len(occupied_qubits_i) > 1:
                # Pick occupied qubit connected to node j
                random.shuffle(occupied_qubits_i)
                qubit_1 = occupied_qubits_i[0]
                j = S[i][qubit_1[0]][qubit_1[1]][2][0]
                # Pick occupied qubit connected to node k!=j, with A_jk=0
                for qubit_2 in occupied_qubits_i:
                    k = S[i][qubit_2[0]][qubit_2[1]][2][0]
                    if ((not k == j) and
                            (S[j][k] == 0)):  # A_jk=0 is equivalent to S_jk=0
                        pairs_list[i] = [qubit_1, qubit_2]
                        break

        # Each node attempts the swap
        for i in node_list:
            # Perform a single swap
            if not pairs_list[i] is None:
                if np.random.rand() < q_swap:
                    S = swap(S, [i, pairs_list[i][0][0], pairs_list[i][0][1]],
                             [i, pairs_list[i][1][0], pairs_list[i][1][1]],
                             ps=p_swap, randomseed=randomseed)
    if p_swap < 1:
        S, count = cutoffs(S, cutoff + 10)  # We need to remove links from failed swaps,
        # which are links with infinite age. We need
        # to give them this age as a placeholder so
        # that all nodes can perform swaps at the
        # same time although in our simulation they
        # do this sequentally
        discard_count += count
    # Remove links that are too long
    if q_swap > 0:
        S, count = remove_long_links(S, max_links_swapped)
        discard_count += count
    # Consume links
    S = consume_fixed_rate(S, p_cons)

    return S, discard_count


def step_protocol_ndsrs(S, p_gen, q_swap_vec, p_swap, p_cons, cutoff, max_links_swapped):
    '''Node-Dependent Single Random Swap (NDSRS) protocol.'''
    n = len(S)

    if not len(q_swap_vec) == n:
        raise ValueError('Parameter q not specified for every node (or for too many nodes)')

    S = advance_time(S)

    S = cutoffs(S, cutoff)

    # Generate links on every available qubit (1 per physical channel)
    S = generate_all_links(S, p_gen)

    # Perform swaps
    if (np.array(q_swap_vec) > 0).any():
        node_list = np.random.permutation(np.arange(n))  # This does not need to be a random permutation

        # Each node chooses a random pair
        pairs_list = [None for _ in range(n)]
        for i in node_list:
            if q_swap_vec[i] == 0:
                continue
            # Find all occupied qubits in node i
            occupied_qubits_i = []
            for z in range(n):
                # If qubit address (i,z,-) exists
                if not S[i][z] == 0:
                    for m, qubit in enumerate(S[i][z]):
                        # If qubit is occupied:
                        if not qubit is None:
                            occupied_qubits_i += [(z, m)]
            if len(occupied_qubits_i) > 1:
                # Pick occupied qubit connected to node j
                random.shuffle(occupied_qubits_i)
                qubit_1 = occupied_qubits_i[0]
                j = S[i][qubit_1[0]][qubit_1[1]][2][0]
                # Pick occupied qubit connected to node k!=j, with A_jk=0
                for qubit_2 in occupied_qubits_i:
                    k = S[i][qubit_2[0]][qubit_2[1]][2][0]
                    if ((not k == j) and
                            (S[j][k] == 0)):  # A_jk=0 is equivalent to S_jk=0
                        pairs_list[i] = [qubit_1, qubit_2]
                        break

        # Each node attempts the swap
        for i in node_list:
            if q_swap_vec[i] == 0:
                continue
            else:
                q_swap = q_swap_vec[i]
            # Perform a single swap
            if not pairs_list[i] is None:
                if np.random.rand() < q_swap:
                    S = swap(S, [i, pairs_list[i][0][0], pairs_list[i][0][1]],
                             [i, pairs_list[i][1][0], pairs_list[i][1][1]],
                             ps=p_swap)

    if p_swap < 1:
        S = cutoffs(S, cutoff + 10)  # We need to remove links from failed swaps,
        # which are links with infinite age. We need
        # to give them this age as a placeholder so
        # that all nodes can perform swaps at the
        # same time although in our simulation they
        # do this sequentally

    # Remove links that are too long
    if (np.array(q_swap_vec) > 0).any():
        S = remove_long_links(S, max_links_swapped)

    # Consume links
    S = consume_fixed_rate(S, p_cons)

    return S


# ---------------------------------------------------------------------------
# ----------------------------- SIMULATIONS ---------------------------------
# ---------------------------------------------------------------------------
def simulation_cd_cfs(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, M, qbits_per_channel, N_samples, total_time,
                      swap_mode, randomseed,
                      progress_bar=None, return_data='avg'):
    ''' ---Inputs---
            · protocol: (str) protocol to be run ('srs' or 'ndsrs' or 'cfs).
            · A:    (array) physical adjacency matrix.
            · p_gen: (float) probability of successful entanglement generation.
            · q_swap:   (float) probability that a swap is attempted. In the 'ndsrs',
                        this has to be a list of length len(A).
            · p_swap:   (float) probability of successful swap.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · M:    (int) maximum swap length.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                    for each physical channel.
            · N_samples:    (int) number of samples.
            · total_time:
            · progress_bar: (str) None, 'notebook', or 'terminal'.
            · return_data:  (str) 'avg' or 'all'.
        ---Outputs---
            '''
    n = len(A)
    if progress_bar == None:
        _tqdm = lambda *x, **kwargs: x[0]
    elif progress_bar == 'notebook':
        _tqdm = tqdmn
    elif progress_bar == 'terminal':
        _tqdm = tqdm
    else:
        raise ValueError('Invalid progress_bar')

    # Calculate physical degrees
    pdegrees = physical_degrees(A)
    # randomseed = randomseed
    if return_data == 'all':
        # Initialize time-dependent node-dependent virtual quantities
        vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # ISSUE: If there is an error, may need to uncomment the four lines below
        # avg_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # avg_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        for sample in _tqdm(range(N_samples), 'Samples', leave=False):
            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'cfs':
                    S = step_protocol_cfs_connectivity_first_swap(S, p_gen, q_swap, p_swap, p_cons, cutoff, M,
                                                                  randomseed, swap_mode)
                    # print(virtual_adjacency_matrix(S))
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    vdeg, vneigh, _, _ = virtual_properties(S, node)
                    vdegrees[node][t][sample] = vdeg
                    vneighs[node][t][sample] = vneigh
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return vdegrees, vneighs, None, None

    if return_data == 'avg':
        # Initialize time-dependent node-dependent virtual quantities
        avg_vdegrees = [[None for _ in range(total_time)] for _ in range(n)]
        avg_vneighs = [[None for _ in range(total_time)] for _ in range(n)]
        std_vdegrees = [[None for _ in range(total_time)] for _ in range(n)]
        std_vneighs = [[None for _ in range(total_time)] for _ in range(n)]

        # First sample
        S = create_qubit_registers(A, qbits_per_channel)
        for t in range(0, total_time):
            if protocol == 'cfs':
                # print("time: " + str(t))
                S, count = step_protocol_cfs_connectivity_first_swap(S, p_gen, q_swap, p_swap, p_cons, cutoff, M,
                                                                     randomseed, swap_mode)
                # print(virtual_adjacency_matrix(S))
            else:
                raise ValueError('Protocol not implemented')
            for node in range(n):
                vdeg, vneigh, vneighborhood, _ = virtual_properties(S, node)
                # print("simulation_cd_for_virtual_neighbors_cfs" + str(vneighborhood))
                avg_vdegrees[node][t] = vdeg
                avg_vneighs[node][t] = vneigh
                std_vdegrees[node][t] = 0
                std_vneighs[node][t] = 0

        # Rest of the samples
        for sample in _tqdm(range(N_samples - 1), 'Samples', leave=False):
            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'cfs':
                    S, count = step_protocol_cfs_connectivity_first_swap(S, p_gen, q_swap, p_swap, p_cons, cutoff, M,
                                                                  randomseed, swap_mode)
                    # print(virtual_adjacency_matrix(S))
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    vdeg, vneigh, _, _ = virtual_properties(S, node)
                    std_vdegrees[node][t] = np.sqrt(std_vdegrees[node][t] ** 2 *
                                                    (sample) / (sample + 1) + sample * (avg_vdegrees[node][t]
                                                                                        - vdeg) ** 2 / (
                                                            sample + 1) ** 2)
                    std_vneighs[node][t] = np.sqrt(std_vneighs[node][t] ** 2 *
                                                   sample / (sample + 1) + sample * (avg_vneighs[node][t]
                                                                                     - vneigh) ** 2 / (sample + 1) ** 2)
                    avg_vdegrees[node][t] = avg_vdegrees[node][t] * sample / (sample + 1) + vdeg / (sample + 1)
                    avg_vneighs[node][t] = avg_vneighs[node][t] * sample / (sample + 1) + vneigh / (sample + 1)

        # Note that we do not store all samples. Instead, we update the mean and
        # the std on the fly. The correctness of this way of updating these values
        # can be checked with the following code:
        ## x_vec = np.random.rand(10)
        ## std = None
        ## mean = None
        ## for idx, x in enumerate(x_vec):
        ##     if std == None and mean == None:
        ##         std = 0
        ##         mean = x
        ##     else:
        ##         N = idx+1
        ##         std = np.sqrt(std**2 * (N-1)/N + (N-1) * (mean-x)**2 / N**2)
        ##         mean = mean * (N-1)/N + x/N
        ## print(np.mean(x_vec),np.std(x_vec))
        ## print(mean, std)

        ## OLD IMPLEMENTATION storing all samples:
        # vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # for sample in _tqdm(range(N_samples), 'Samples', leave=False):
        #     S = create_qubit_registers(A, qbits_per_channel)
        #     for t in range(0,total_time):
        #         if protocol=='rprs':
        #             S = step_protocol_rprs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
        #         elif protocol=='srs':
        #             S = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
        #         else:
        #             raise ValueError('Protocol not implemented')
        #         for node in range(n):
        #             vdeg, vneigh, _, _ = virtual_properties(S, node)
        #             vdegrees[node][t][sample] = vdeg
        #             vneighs[node][t][sample] = vneigh
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs


def simulation_cd(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, M, qbits_per_channel, N_samples, total_time,
                  progress_bar=None, return_data='avg'):
    ''' ---Inputs---
            · protocol: (str) protocol to be run ('srs' or 'ndsrs' or 'cfs).
            · A:    (array) physical adjacency matrix.
            · p_gen: (float) probability of successful entanglement generation.
            · q_swap:   (float) probability that a swap is attempted. In the 'ndsrs',
                        this has to be a list of length len(A).
            · p_swap:   (float) probability of successful swap.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · M:    (int) maximum swap length.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                    for each physical channel.
            · N_samples:    (int) number of samples.
            · total_time:
            · progress_bar: (str) None, 'notebook', or 'terminal'.
            · return_data:  (str) 'avg' or 'all'.
        ---Outputs---
            '''
    n = len(A)
    if progress_bar == None:
        _tqdm = lambda *x, **kwargs: x[0]
    elif progress_bar == 'notebook':
        _tqdm = tqdmn
    elif progress_bar == 'terminal':
        _tqdm = tqdm
    else:
        raise ValueError('Invalid progress_bar')

    # Calculate physical degrees
    pdegrees = physical_degrees(A)
    randomseed = np.random.seed()
    if return_data == 'all':
        # Initialize time-dependent node-dependent virtual quantities
        vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # ISSUE: If there is an error, may need to uncomment the four lines below
        # avg_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # avg_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        for sample in _tqdm(range(N_samples), 'Samples', leave=False):

            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'srs':
                    S, count = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
                elif protocol == 'ndsrs':
                    S = step_protocol_ndsrs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    vdeg, vneigh, _, _ = virtual_properties(S, node)
                    assert vdeg == vneigh
                    vdegrees[node][t][sample] = vdeg
                    vneighs[node][t][sample] = vneigh
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return vdegrees, vneighs, None, None

    if return_data == 'avg':
        # Initialize time-dependent node-dependent virtual quantities
        avg_vdegrees = [[None for _ in range(total_time)] for _ in range(n)]
        avg_vneighs = [[None for _ in range(total_time)] for _ in range(n)]
        std_vdegrees = [[None for _ in range(total_time)] for _ in range(n)]
        std_vneighs = [[None for _ in range(total_time)] for _ in range(n)]

        # First sample
        S = create_qubit_registers(A, qbits_per_channel)
        for t in range(0, total_time):
            if protocol == 'srs':
                S, count = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M, randomseed)
            elif protocol == 'ndsrs':
                S = step_protocol_ndsrs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
            else:
                raise ValueError('Protocol not implemented')
            for node in range(n):
                vdeg, vneigh, _, _ = virtual_properties(S, node)
                avg_vdegrees[node][t] = vdeg
                avg_vneighs[node][t] = vneigh
                std_vdegrees[node][t] = 0
                std_vneighs[node][t] = 0

        # Rest of the samples
        for sample in _tqdm(range(N_samples - 1), 'Samples', leave=False):
            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'srs':
                    S, count = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M, randomseed)
                elif protocol == 'ndsrs':
                    S = step_protocol_ndsrs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    vdeg, vneigh, _, _ = virtual_properties(S, node)
                    std_vdegrees[node][t] = np.sqrt(std_vdegrees[node][t] ** 2 *
                                                    (sample) / (sample + 1) + sample * (avg_vdegrees[node][t]
                                                                                        - vdeg) ** 2 / (
                                                            sample + 1) ** 2)
                    std_vneighs[node][t] = np.sqrt(std_vneighs[node][t] ** 2 *
                                                   sample / (sample + 1) + sample * (avg_vneighs[node][t]
                                                                                     - vneigh) ** 2 / (sample + 1) ** 2)
                    avg_vdegrees[node][t] = avg_vdegrees[node][t] * sample / (sample + 1) + vdeg / (sample + 1)
                    avg_vneighs[node][t] = avg_vneighs[node][t] * sample / (sample + 1) + vneigh / (sample + 1)

        # Note that we do not store all samples. Instead, we update the mean and
        # the std on the fly. The correctness of this way of updating these values
        # can be checked with the following code:
        ## x_vec = np.random.rand(10)
        ## std = None
        ## mean = None
        ## for idx, x in enumerate(x_vec):
        ##     if std == None and mean == None:
        ##         std = 0
        ##         mean = x
        ##     else:
        ##         N = idx+1
        ##         std = np.sqrt(std**2 * (N-1)/N + (N-1) * (mean-x)**2 / N**2)
        ##         mean = mean * (N-1)/N + x/N
        ## print(np.mean(x_vec),np.std(x_vec))
        ## print(mean, std)

        ## OLD IMPLEMENTATION storing all samples:
        # vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # for sample in _tqdm(range(N_samples), 'Samples', leave=False):
        #     S = create_qubit_registers(A, qbits_per_channel)
        #     for t in range(0,total_time):
        #         if protocol=='rprs':
        #             S = step_protocol_rprs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
        #         elif protocol=='srs':
        #             S = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
        #         else:
        #             raise ValueError('Protocol not implemented')
        #         for node in range(n):
        #             vdeg, vneigh, _, _ = virtual_properties(S, node)
        #             vdegrees[node][t][sample] = vdeg
        #             vneighs[node][t][sample] = vneigh
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs


def simulation_cd_for_virtual_neighbors(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, M, qbits_per_channel,
                                        N_samples, total_time,
                                        randomseed, progress_bar=None, return_data='all'):
    ''' ---Inputs---
            · protocol: (str) protocol to be run ('srs' or 'ndsrs' or cfs).
            · A:    (array) physical adjacency matrix.
            · p_gen: (float) probability of successful entanglement generation.
            · q_swap:   (float) probability that a swap is attempted. In the 'ndsrs',
                        this has to be a list of length len(A).
            · p_swap:   (float) probability of successful swap.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · M:    (int) maximum swap length.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                    for each physical channel.
            · N_samples:    (int) number of samples.
            · total_time:
            · progress_bar: (str) None, 'notebook', or 'terminal'.
            · return_data:  (str) 'avg' or 'all'.
        ---Outputs---
            '''
    n = len(A)
    if progress_bar == None:
        _tqdm = lambda *x, **kwargs: x[0]
    elif progress_bar == 'notebook':
        _tqdm = tqdmn
    elif progress_bar == 'terminal':
        _tqdm = tqdm
    else:
        raise ValueError('Invalid progress_bar')

    # np.random.seed(randomseed)
    # Calculate physical degrees
    pdegrees = physical_degrees(A)

    if return_data == 'all':
        # Initialize time-dependent node-dependent virtual quantities
        vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighborhoods_all_links = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # ISSUE: If there is an error, may need to uncomment the four lines below
        # avg_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # avg_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        for sample in _tqdm(range(N_samples), 'Samples', leave=False):

            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'srs':
                    S, count = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M, randomseed)
                elif protocol == 'ndsrs':
                    S = step_protocol_ndsrs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    # Todo:
                    vneighborhood_count = [0] * n
                    vdeg, vneigh, vneighborhood, vneigh_links = virtual_properties(S, node, vneighborhood_count)
                    vdegrees[node][t][sample] = vdeg
                    vneighs[node][t][sample] = vneigh
                    # print("simulation_cd_for_virtual_neighbors_cfs" + str(vneighborhood))
                    vneighborhoods[node][t][sample] = vneighborhood
                    vneighborhoods_all_links[node][t][sample] = vneighborhood_count

                    # vdeg, vneigh, vneighborhood, vneigh_links = virtual_properties(S, node)
                    # vdegrees[node][t][sample] = vdeg
                    # vneighs[node][t][sample] = vneigh
                    # vneighborhoods[node][t][sample] = vneighborhood
                    # vneighborhoods_all_links[node][t][sample] = vneigh_links
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees
        # = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return vdegrees, vneighs, vneighborhoods, vneighborhoods_all_links


def simulation_cd_for_virtual_neighbors_cfs(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, M, qbits_per_channel,
                                            N_samples, total_time, swap_mode,
                                            randomseed, progress_bar=None, return_data='all'):
    ''' ---Inputs---
            · protocol: (str) protocol to be run ('srs' or 'ndsrs' or cfs).
            · A:    (array) physical adjacency matrix.
            · p_gen: (float) probability of successful entanglement generation.
            · q_swap:   (float) probability that a swap is attempted. In the 'ndsrs',
                        this has to be a list of length len(A).
            · p_swap:   (float) probability of successful swap.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · M:    (int) maximum swap length.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                    for each physical channel.
            · N_samples:    (int) number of samples.
            · total_time:
            · progress_bar: (str) None, 'notebook', or 'terminal'.
            · return_data:  (str) 'avg' or 'all'.
        ---Outputs---
            '''
    n = len(A)
    if progress_bar == None:
        _tqdm = lambda *x, **kwargs: x[0]
    elif progress_bar == 'notebook':
        _tqdm = tqdmn
    elif progress_bar == 'terminal':
        _tqdm = tqdm
    else:
        raise ValueError('Invalid progress_bar')

    # np.random.seed(randomseed)
    # Calculate physical degrees
    pdegrees = physical_degrees(A)

    if return_data == 'all':
        # Initialize time-dependent node-dependent virtual quantities
        vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighborhoods_all_links = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        # ISSUE: If there is an error, may need to uncomment the four lines below
        # avg_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # avg_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        for sample in _tqdm(range(N_samples), 'Samples', leave=False):
            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'cfs':
                    S, count = step_protocol_cfs_connectivity_first_swap(S, p_gen, q_swap, p_swap, p_cons, cutoff, M,
                                                                         randomseed, swap_mode)
                    # print("time: " + str(t))
                    # print(virtual_adjacency_matrix(S))
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    # Todo:
                    vneighborhood_count = [0] * n
                    vdeg, vneigh, vneighborhood, vneigh_links = virtual_properties(S, node, vneighborhood_count)
                    vdegrees[node][t][sample] = vdeg
                    vneighs[node][t][sample] = vneigh
                    # print("simulation_cd_for_virtual_neighbors_cfs" + str(vneighborhood))
                    vneighborhoods[node][t][sample] = vneighborhood
                    vneighborhoods_all_links[node][t][sample] = vneighborhood_count
                    # vneighborhood_count[node][t][sample] = vneighborhood_count
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees
        # = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return vdegrees, vneighs, vneighborhoods, vneighborhoods_all_links


# ---------------------------------------------------------------------------
# ------------------------- PERFORMANCE METRICS -----------------------------
# ---------------------------------------------------------------------------
def virtual_properties(S, node, vneighborhood_count = None):
    n = len(S)

    vdeg = 0  # Virtual degree
    vneigh = 0  # Virtual neighborhood size
    vneighborhood = set()  # Virtual neighborhood
    vneigh_links = [[] for _ in range(n)]  # Element i contains the IDs of qubits
    # that share an entangled link with node i
    # vneigh_all = [[] for _ in range(n)]
    # vneighborhood_count = [0] * n

    for j in range(n):
        # If qubit address (node,j,-) exists:
        if not S[node][j] == 0:
            for m, qubit in enumerate(S[node][j]):
                # If qubit is occupied:
                if not qubit is None:
                    vdeg += 1
                    vneigh_links[S[node][j][m][2][0]] += [(node, j, m)]
# % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
#             % S[i][j][m][0]: age of the qubit (<cutoff+1).
#             % S[i][j][m][1]: number of times its link was involved in swaps (<M).
#             % S[i][j][m][2]: address of the qubit with which it is entangled.
#             % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
#             % S[i][j]=0.0, if nodes i and j do not share a physical link.
#             % S[i][j][m]=None, if qubit (i,j,m) is unused.
#                     assert vneigh_links[S[node][j][m][2][0]] == j
                    if not S[node][j][m][2][0] in vneighborhood:
                        vneighborhood.add(S[node][j][m][2][0])
                        vneigh += 1
                    if vneighborhood_count:
                        vneighborhood_count[S[node][j][m][2][0]] += 1
                        # print(S[node][j][m][2][0])
                        # print((node, j, m))
                        # Todo:
    # print("virtual_properties: node" + str(node) + " " + str(vneighborhood))
    # if vneighborhood_count:
    return vdeg, vneigh, list(vneighborhood), vneigh_links

# backup
# def virtual_properties(S, node):
#     n = len(S)
#
#     vdeg = 0  # Virtual degree
#     vneigh = 0  # Virtual neighborhood size
#     vneighborhood = set()  # Virtual neighborhood
#     vneigh_links = [[] for _ in range(n)]  # Element i contains the IDs of qubits
#     # that share an entangled link with node i
#     # vneigh_all = [[] for _ in range(n)]
#
#     for j in range(n):
#         # If qubit address (node,j,-) exists:
#         if not S[node][j] == 0:
#             for m, qubit in enumerate(S[node][j]):
#                 # If qubit is occupied:
#                 if not qubit is None:
#                     vdeg += 1
#                     vneigh_links[S[node][j][m][2][0]] += [(node, j, m)]
# # % S[i][j][m] contains the info about the qubit with address (i,j,m), which is the qubit held by node i in register (j,m), meaning that it was initially used to generate an elementary link with node j.
# #             % S[i][j][m][0]: age of the qubit (<cutoff+1).
# #             % S[i][j][m][1]: number of times its link was involved in swaps (<M).
# #             % S[i][j][m][2]: address of the qubit with which it is entangled.
# #             % address: [a, b ,c] a: location (node a), b: 与node b 相连, c: 第c个通信比特
# #             % S[i][j]=0.0, if nodes i and j do not share a physical link.
# #             % S[i][j][m]=None, if qubit (i,j,m) is unused.
# #                     assert vneigh_links[S[node][j][m][2][0]] == j
#                     if not S[node][j][m][2][0] in vneighborhood:
#                         vneighborhood.add(S[node][j][m][2][0])
#                         vneigh += 1
#                         # print(S[node][j][m][2][0])
#                         # print((node, j, m))
#     # print("virtual_properties: node" + str(node) + " " + str(vneighborhood))
#     return vdeg, vneigh, list(vneighborhood), vneigh_links



def virtual_links_ij(S, node_i, node_j):
    n = len(S)

    wij = 0  # Number of virtual links between both nodes
    for k in range(n):
        if not S[node_i][k] == 0:  # If qubit register node_i-k exists
            for m, qubit in enumerate(S[node_i][k]):
                if not qubit is None:  # If qubit is occupied
                    if qubit[2][0] == node_j:  # If qubit linked to node_j
                        wij += 1
    return wij


def total_qubits_occupied(S):
    '''Calculates the total number of qubits that are in use (i.e., that are
        holding one half of an entangled link).'''
    occupied_qubits = 0
    n = len(S)
    for node1 in range(n):
        for node2 in range(n):
            # If there is a physical channel between node1 and node2:
            if not S[node1][node2] == 0:
                for qubit in S[node1][node2]:
                    # If qubit is occupied:
                    if not qubit is None:
                        occupied_qubits += 1
    return occupied_qubits


def total_qubits_occupied_node(S, node):
    '''Calculates the total number of qubits that are in use (i.e., that are
        holding one half of an entangled link) at a given node.'''
    occupied_qubits = 0
    n = len(S)
    for node2 in range(n):
        occupied_qubits += virtual_links_ij(S, node, node2)
    return occupied_qubits


# ---------------------------------------------------------------------------
# ------------------------- AUXILIARY FUNCTIONS -----------------------------
# ---------------------------------------------------------------------------
def random_pairs(my_list):
    random.shuffle(my_list)
    return [(my_list[i], my_list[i + 1]) for i in np.arange(0, len(my_list) - 1, 2)]


def find_steady_state(avg_array, error, window, window_type='shrinking'):
    '''Find the (constant) steady state. Let us consider a (noisy) function
        y(x). The input variable `avg_array` and `error_array` contain 
        N samples of the average value and the error (e.g., the standard
        error) of the function for increasing and equispaced values of x. 
        We assume the steady state is reached when:
            - SLIDING WINDOW: a window of `window` samples moves, increasing
                x, until the stopping condition is met. Stopping condition:
                all the intervals of confidence of the samples within the
                window overlap.
            - SHRINKING ANY OVERLAP: the window initially includes all samples,
                and it progressively includes less samples (ignoring the
                first ones), until the stopping condition is met. 
                Stopping condition: all the intervals of confidence of the
                samples within the window overlap. 
            - SHRINKING: the window initially includes all samples,
                and it progressively includes less samples (ignoring the
                first ones), until the stopping condition is met. 
                Stopping condition: all overlaps are smaller than 1.5
                times the error (as indicated by our algorithm in the draft). 
        ---Inputs---
            · avg_array: (array) equispaced average values of the function.
            · error_array: (array) equispaced standard errors of the
                            function.
            · window:   (int) number of samples with which the absolute
                        difference is calculated for each sample.'''
    window = int(window)
    if window_type == 'sliding':
        # SLIDING WINDOW
        for i, elem in enumerate(avg_array):
            if i <= len(avg_array) - window:
                upper_bounds = [avg_array[j] + error_array[j] for j in range(i, i + window)]
                lower_bounds = [avg_array[j] - error_array[j] for j in range(i, i + window)]
                # If all intervals of confidence overlap, we found the steady state
                if min(upper_bounds) > max(lower_bounds):
                    return i
        # If no steady state was found, return None
        return None
    elif window_type == 'shrinking_any_overlap':
        # SHRINKING WINDOW
        upper_bounds = np.array(avg_array) + np.array(error_array)
        lower_bounds = np.array(avg_array) - np.array(error_array)
        for i, elem in enumerate(avg_array):
            if i <= len(avg_array) - window:
                upbounds = upper_bounds[i:]
                lowbounds = lower_bounds[i:]
                # If all intervals of confidence overlap, we found the steady state
                if min(upbounds) > max(lowbounds):
                    return i
            else:
                # If no steady state was found, return None
                return None
    elif window_type == 'shrinking':
        # SHRINKING WINDOW, ASSUMING THE ERROR ARRAY IS A SINGLE NUMBER
        threshold = 1.5 * error
        avg_array = np.array(avg_array)
        overlaps = []

        # Calculate overlaps in the minimum window
        for i in range(1, window + 1):
            for j in range(1, window + 1):
                overlaps += [(2 * error
                              - np.abs(avg_array[-i] - avg_array[-j]))]
        if any(overlaps < threshold):
            # Not even the smallest window seems to be in steady state
            return None

        # Keep increasing the window
        for i in range(window + 1, len(avg_array) + 1):
            # for i in tqdmn(range(window+1, len(avg_array)+1), 'Steady state...', leave=False):
            new_overlaps = []
            # Calculate overlap with all the previous samples
            for j in range(1, i):
                new_overlaps += [(2 * error
                                  - np.abs(avg_array[-i] - avg_array[-j]))]
            if any(new_overlaps < threshold):
                # We found the end of the steady state
                return int(len(avg_array) - i + 1)

        return int(len(avg_array) - i)


def find_m_neighborhood(A, start_node, m):
    '''Calculate the m-neighborhood of start_node, i.e., the set of
        nodes that are m edges away from start_node. We use the
        breadth-first search algorithm. Function originally provided
        by chatGPT (but tested by myself!).
        using a 
        ---Inputs---
            · A:    (array) adjacency matrix.
            · start_node:   (int) number of samples with which the absolute
                        difference is calculated for each sample.
            · m:    (float) distance between start_node and other nodes.'''
    n = len(A)
    visited = [False] * n
    visited[start_node] = True
    distances = [0] * n
    distances[start_node] = 0
    q = Queue()
    q.put(start_node)
    m_neighborhood = set()

    while not q.empty():
        node = q.get()
        if distances[node] == m:
            m_neighborhood.add(node)
        if distances[node] > m:
            break
        for neighbor in range(n):
            if A[node][neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = distances[node] + 1
                q.put(neighbor)

    return m_neighborhood


def find_pareto_region_cd(users, varying_array, varying_param, protocol, data_type, topology, n, p_gen, q_swap, p_swap,
                          p_cons, cutoff, max_links_swapped, qbits_per_channel, N_samples, total_time, randomseed):
    '''
        ---Inputs---
            · ...
            · users:    (list of ints) nodes that will be analyzed.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · ...
        ---Outputs---'''

    #####################
    ## LOAD DATA ##
    #####################

    print('WARNING: presence of steady state not checked')

    data = [[] for _ in range(len(varying_array))]
    data_std = [[] for _ in range(len(varying_array))]

    for data_idx, varying_value in enumerate(varying_array):
        if varying_param == 'p_gen':
            p_gen = varying_value
        elif varying_param == 'q_swap':
            q_swap = varying_value
        else:
            raise ValueError('varying_param has an invalid value.')
        _data = load_data_cd(protocol, data_type, topology, n, p_gen, q_swap,
                             p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel,
                             N_samples, total_time, randomseed)

        # ANALYZE STEADY STATE
        # ...

        # Save data
        for user_idx, user in enumerate(users):
            data[data_idx] += [_data['avg_vneighs'][users[user_idx]][-1]]
            data_std[data_idx] += [2 * _data['std_vneighs'][users[user_idx]][-1] / np.sqrt(N_samples)]

    #####################
    ## PARETO ##
    #####################
    assert len(data) == len(varying_array)
    for data_point in data:
        assert len(data_point) == len(users)
    return find_pareto_region(data)


def find_pareto_region(data):
    '''data is a list of lists. Each sublist is an N-dimensional data point.'''
    pareto_frontier = set()

    for candidate_idx, candidate in enumerate(data):
        candidate_is_pareto = True
        # Check if any other data point dominates the candidate
        for data_idx, data_point in enumerate(data):
            if data_idx == candidate_idx:
                continue
            if (np.array(candidate) < np.array(data_point)).all():
                candidate_is_pareto = False
        if candidate_is_pareto:
            pareto_frontier.add(candidate_idx)
    return pareto_frontier


def time_correlation(array, k):
    '''Calculate the time correlation function with argument k:
        c(k) = average(array[i]*array[i+k]) - average(array)^2.
        If k=0, this function returns the variance of the array.'''
    # If k is too large compared to the array length, the averages
    # are not reliable
    if k > len(array) / 2:
        return None

    diff_array = [array[idx] * array[idx + k] for idx in range(len(array) - k)]
    term1 = np.mean(diff_array)
    term2 = np.mean(array) ** 2

    return term1 - term2


def correlation_time(array):
    '''Estimate the integrated correlation time:
        tau = 0.5 * sum_{k=-inf}^{+inf} c(k)/c(0),
        where c() is the time correlation function.'''
    max_k = len(array) // 2
    sum_k = sum([time_correlation(array, k) for k in range(max_k)])
    return 0.5 * sum_k / time_correlation(array, 0)


# ---------------------------------------------------------------------------
# --------------------------- THEORY RESULTS --------------------------------
# ---------------------------------------------------------------------------
def theory_virtual_noswaps(physical_degree, p_gen, p_cons, cutoff, qbits_per_channel):
    ''' ---Inputs---
            · physical_degree:  (int) physical degree of node i.
            · p_gen: (float) probability of successful entanglement generation.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                for each physical channel.
        ---Outputs---
            · avg_vneigh:   (float) expected number of virtual neighbors of node i.
            · avg_vdeg:     (float) expected virtual degree of node i.'''
    gamma = min(qbits_per_channel, cutoff)

    # Probability of no links between two physical neighbors in steady state
    # In the manuscript, this is denoted by variable $\pi_0$
    nolinks_ss = (p_gen - p_cons) / ((1 - p_gen) *
                                     (p_gen * (p_gen * (1 - p_cons) / (p_cons * (1 - p_gen))) ** gamma - p_cons))

    # Expected number of links between two physical neighbors in steady state
    if p_gen == p_cons:  # To avoid singularities
        p_cons += 1e-9
    if p_gen == 1:
        p_gen -= 1e-10
    if p_cons == 1:
        p_cons -= 1e-10
    rho = (p_gen * (1 - p_cons)) / (p_cons * (1 - p_gen))
    avg_links = (p_gen * (gamma * (p_gen - p_cons) * rho ** gamma
                          + p_cons * (1 - p_cons) * (1 - rho ** gamma))
                 / ((p_gen - p_cons) * (p_gen * rho ** gamma - p_cons)))

    # Expected virtual degree
    avg_vdeg = physical_degree * avg_links

    # Expected number of virtual neighbors
    avg_vneigh = physical_degree * (1 - nolinks_ss)

    return avg_vdeg, avg_vneigh, avg_links, nolinks_ss


# ---------------------------------------------------------------------------
# ------------------------------- PLOTS -------------------------------------
# ---------------------------------------------------------------------------
def plot_avgs(users, protocol, data_type, topology, n, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped,
              qbits_per_channel, N_samples, total_time, randomseed, nd_label=None, avg_vdegs_theory=None,
              avg_vneighs_theory=None, physical_degrees=None, steady_state_window=None, steady_state_force_find=False,
              dark=False, legend='nodes', save=False, x_cm=8, y_cm=5, fontsize=8, xlimits=None, ylimits_deg=None,
              ylimits_neigh=None, num_y_ticks_vdeg=4, num_y_ticks_vneigh=4, **kwargs):
    '''Plot average virtual metrics over time'''

    #####################
    ## LOAD DATA ##
    #####################
    sim_data = load_data_cd(protocol, data_type, topology, n, p_gen, q_swap,
                            p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel,
                            N_samples, total_time, randomseed, nd_label=nd_label)
    avg_vdegs_sim = sim_data['avg_vdegrees']
    avg_vneighs_sim = sim_data['avg_vneighs']
    std_vdegs_sim = sim_data['std_vdegrees']
    std_vneighs_sim = sim_data['std_vneighs']

    # Find steady state
    if steady_state_window is not None:
        avg_vdegs_ss_idx = [None for _ in users]
        avg_vneighs_ss_idx = [None for _ in users]
        number_of_nodes = len(avg_vdegs_sim)
        for user_idx, user in enumerate(users):
            if steady_state_force_find:  # Ensure that we find a steady state
                # by making errors progressively large
                error_increase = 1
                while avg_vdegs_ss_idx[user_idx] is None:
                    error_vdeg = error_increase * physical_degrees[user] * qbits_per_channel / np.sqrt(N_samples)
                    avg_vdegs_ss_idx[user_idx] = find_steady_state(avg_vdegs_sim[user], error_vdeg,
                                                                   steady_state_window,
                                                                   window_type='shrinking')
                    error_increase = error_increase + 0.1
                error_increase = 1
                while avg_vneighs_ss_idx[user_idx] is None:
                    error_vneigh = error_increase * min(physical_degrees[user] * qbits_per_channel,
                                                        number_of_nodes) / np.sqrt(N_samples)
                    avg_vneighs_ss_idx[user_idx] = find_steady_state(avg_vneighs_sim[user], error_vneigh,
                                                                     steady_state_window,
                                                                     window_type='shrinking')
                    error_increase = error_increase + 0.1
            else:
                error_vdeg = physical_degrees[user] * qbits_per_channel / np.sqrt(N_samples)
                avg_vdegs_ss_idx[user_idx] = find_steady_state(avg_vdegs_sim[user], error_vdeg,
                                                               steady_state_window,
                                                               window_type='shrinking')
                error_vneigh = min(physical_degrees[user] * qbits_per_channel,
                                   number_of_nodes) / np.sqrt(N_samples)
                avg_vneighs_ss_idx[user_idx] = find_steady_state(avg_vneighs_sim[user], error_vneigh,
                                                                 steady_state_window,
                                                                 window_type='shrinking')

    #####################
    ## PLOTS ##
    #####################
    cmap = plt.cm.get_cmap('viridis')
    if dark == True:
        plt.style.use('dark_background')
        main_color = 'w'
        if len(users) == 3:
            colors = [cmap(0.3), cmap(0.7), cmap(0.99)]
        else:
            colors = [cmap(i / len(users)) for i in range(len(users))]
    else:
        plt.style.use('default')
        main_color = 'k'
        if len(users) == 3:
            colors = [cmap(0), cmap(0.45), cmap(0.8)]
        else:
            colors = [cmap(i / len(users)) for i in range(len(users))]

    total_time = len(avg_vdegs_sim[0])

    # Virtual degree
    if True:
        fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))
        for idx, node in enumerate(users):
            if legend == 'nodes':
                label = 'Node %d' % node
            elif legend == 'levels':
                label = 'Level %d node' % node

            # Average value
            avg_vdegs_sim_i = np.array([0] + list(avg_vdegs_sim[node]))  # Add 0 at
            # start because simulation does not save starting value
            std_vdegs_sim_i = np.array([0] + list(std_vdegs_sim[node]))  # Add 0 at
            # start because simulation does not save starting value
            plt.plot(avg_vdegs_sim_i, color=colors[idx], label=label)

            # Interval of confidence
            stderror_vdegs = 2 * std_vdegs_sim_i / np.sqrt(
                N_samples)  # 3*min(cutoff,qbits_per_channel)/np.sqrt(N_samples)
            ax.fill_between(range(total_time + 1),
                            avg_vdegs_sim_i + stderror_vdegs,
                            avg_vdegs_sim_i - stderror_vdegs,
                            color=colors[idx], alpha=0.3)

            # Theory
            if avg_vdegs_theory:
                if idx == len(users) - 1:
                    label = 'Theory'
                else:
                    label = ''
                plt.plot([0, total_time], [avg_vdegs_theory[node], avg_vdegs_theory[node]],
                         '--', color=main_color, label=label)

            # Find steady state
            if steady_state_window is not None:
                ss_idx = avg_vdegs_ss_idx[idx]
                if ss_idx is not None:
                    plt.scatter([ss_idx], [avg_vdegs_sim_i[ss_idx]],
                                color=colors[idx])
        if xlimits == None:
            plt.xlim(0, total_time)
        else:
            plt.xlim(xlimits[0], xlimits[1])
        if ylimits_deg == None:
            plt.gca().set_ylim(bottom=0)
        else:
            plt.ylim(ylimits_deg[0], ylimits_deg[1])
        plt.xlabel('Time', fontsize=fontsize)
        plt.ylabel('$k_i$', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # Set the y-axis to have only three major ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_y_ticks_vdeg))
        if len(users) < 6:
            plt.legend(fontsize=fontsize, loc='best', ncol=2)
        if save:
            filename = 'figs/avg_vdegrees_%s_%s_n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f' \
                       '-p_cons%.3f-cutoff%d-max_links_swapped%d-qbits_per_channel%d' \
                       '-N_samples%d-total_time%d-randomseed%s-maxstd%.5f' \
                       '-maxstderr%.5f.pdf' % (protocol, topology,
                                               n, p_gen, q_swap, p_swap,
                                               p_cons, cutoff, max_links_swapped,
                                               qbits_per_channel, N_samples, total_time,
                                               randomseed,
                                               np.max(std_vdegs_sim), np.max(2 * std_vdegs_sim / np.sqrt(N_samples)))
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            print('Maximum std: %.3f' % np.max(std_vdegs_sim))
            print('Maximum standard error: %.3f' % np.max(2 * std_vdegs_sim / np.sqrt(N_samples)))

    # Virtual neighborhood
    if True:
        fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))
        for idx, node in enumerate(users):
            if legend == 'nodes':
                label = 'Node %d' % node
            elif legend == 'levels':
                label = 'Level %d node' % node

            # Average value
            avg_vneighs_sim_i = np.array([0] + list(avg_vneighs_sim[node]))  # Add 0 at
            # start because simulation does not save starting value
            std_vneighs_sim_i = np.array([0] + list(std_vneighs_sim[node]))  # Add 0 at
            # start because simulation does not save starting value
            plt.plot(avg_vneighs_sim_i, color=colors[idx], label=label)

            # Interval of confidence
            stderror_vneighs = 2 * std_vneighs_sim_i / np.sqrt(N_samples)  # 3/np.sqrt(N_samples)
            ax.fill_between(range(total_time + 1),
                            avg_vneighs_sim_i + stderror_vneighs,
                            avg_vneighs_sim_i - stderror_vneighs,
                            color=colors[idx], alpha=0.2)

            # Theory
            if avg_vneighs_theory:
                if idx == len(users) - 1:
                    label = 'Theory'
                else:
                    label = ''
                plt.plot([0, total_time], [avg_vneighs_theory[node],
                                           avg_vneighs_theory[node]],
                         '--', color=main_color, label=label)

            # Find steady state
            if steady_state_window is not None:
                ss_idx = avg_vneighs_ss_idx[idx]
                if ss_idx is not None:
                    plt.scatter([ss_idx], [avg_vneighs_sim_i[ss_idx]],
                                color=colors[idx])

            # # Find steady state
            # if steady_state_window is not None:
            #     if avg_vneighs_ss_idx is not None:
            #         plt.scatter([avg_vneighs_ss_idx],[avg_vneighs_sim_i[i] for i in avg_vneighs_ss_idx],
            #                     color=colors[idx])

        if xlimits == None:
            plt.xlim(0, total_time)
        else:
            plt.xlim(xlimits[0], xlimits[1])
        if ylimits_neigh == None:
            plt.gca().set_ylim(bottom=0)
        else:
            plt.ylim(ylimits_neigh[0], ylimits_neigh[1])
        plt.xlabel('Time', fontsize=fontsize)
        plt.ylabel('$v_i$', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # Set the y-axis to have only three major ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_y_ticks_vneigh))
        if len(users) < 6:
            plt.legend(fontsize=fontsize, loc='best', ncol=2)
        if save:
            filename = 'figs/avg_vneighs_%s_%s_n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f' \
                       '-p_cons%.3f-cutoff%d-max_links_swapped%d-qbits_per_channel%d' \
                       '-N_samples%d-total_time%d-randomseed%s-maxstd%.5f' \
                       '-maxstderr%.5f.pdf' % (protocol, topology, n, p_gen, q_swap, p_swap,
                                               p_cons, cutoff, max_links_swapped,
                                               qbits_per_channel, N_samples, total_time, randomseed,
                                               np.max(std_vneighs_sim),
                                               np.max(2 * std_vneighs_sim / np.sqrt(N_samples)))
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        print('Maximum std: %.3f' % np.max(std_vneighs_sim))
        print('Maximum standard error: %.3f' % np.max(2 * std_vneighs_sim / np.sqrt(N_samples)))


def plot_pareto_singleprotocol_2users(users, varying_array, varying_param, protocol, data_type, topology, n, p_gen,
                                      q_swap, p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel, N_samples,
                                      total_time, randomseed, physical_degrees=None, steady_state_window=None,
                                      steady_state_force_find=False, label_x=None, label_y=None, constraint_x=None,
                                      constraint_y=None, x_cm=8, y_cm=5, fontsize=8, num_x_ticks=3, num_y_ticks=3,
                                      dark=False, save=False, **kwargs):
    '''
        ---Inputs---
            · ...
            · users:    (list of ints) nodes that will be analyzed.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · ...
        ---Outputs---'''

    assert len(users) == 2, 'Number of users should be 2'

    if protocol == 'ndsrs':
        raise ValueError('Function unavailable for node-dependent protocols')
    elif protocol not in ['srs', 'rprs']:
        raise ValueError('Unknown protocol')

    # #####################
    # ## LOAD DATA ##
    # #####################

    data_x = []
    data_x_std = []
    data_y = []
    data_y_std = []

    if steady_state_window is None:
        # If no window is specified, we assume the steady state was reached
        print('WARNING: presence of steady state not checked')

    for data_idx, varying_value in enumerate(varying_array):
        if varying_param == 'p_gen':
            p_gen = varying_value
        elif varying_param == 'q_swap':
            q_swap = varying_value
        else:
            raise ValueError('varying_param has an invalid value.')
        _data = load_data_cd(protocol, data_type, topology, n, p_gen, q_swap,
                             p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel,
                             N_samples, total_time, randomseed)

        # ANALYZE STEADY STATE
        number_of_nodes = len(_data['avg_vneighs'])
        # Find steady state
        steady_state_achieved = True
        if steady_state_window is None:
            # If no window is specified, we assume the steady state was reached
            pass
        else:
            for user in users:

                if steady_state_force_find:  # Ensure that we find a steady state
                    # by making errors progressively large
                    vneighs_ss_idx = None
                    error_increase = 1
                    while vneighs_ss_idx is None:
                        error_vneigh = error_increase * min(physical_degrees[user] * qbits_per_channel,
                                                            number_of_nodes) / np.sqrt(N_samples)
                        vneighs_ss_idx = find_steady_state(_data['avg_vneighs'][user], error_vneigh,
                                                           steady_state_window,
                                                           window_type='shrinking')
                        error_increase = error_increase + 0.1
                else:
                    error_vneigh = min(physical_degrees[user] * qbits_per_channel,
                                       number_of_nodes) / np.sqrt(N_samples)
                    vneighs_ss_idx = find_steady_state(_data['avg_vneighs'][user], error_vneigh,
                                                       steady_state_window,
                                                       window_type='shrinking')
                if vneighs_ss_idx is None:
                    steady_state_achieved = False

        # Save data
        if steady_state_achieved:
            data_x += [_data['avg_vneighs'][users[0]][-1]]
            data_y += [_data['avg_vneighs'][users[1]][-1]]
            data_x_std += [2 * _data['std_vneighs'][users[0]][-1] / np.sqrt(N_samples)]
            data_y_std += [2 * _data['std_vneighs'][users[1]][-1] / np.sqrt(N_samples)]
        else:
            data_x += [None]
            data_y += [None]
            data_x_std += [None]
            data_y_std += [None]

    #####################
    ## PARETO ##
    #####################
    data_pareto = [[data_x[i], data_y[i]] for i in range(len(data_x))]
    pareto_frontier = find_pareto_region(data_pareto)

    #####################
    ## PLOTS ##
    #####################
    if dark == True:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    color = 'k'
    fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))

    # Plot constraints
    x0 = min(data_x) - abs(min(data_x))
    y0 = min(data_y) - abs(min(data_y))
    x1 = max(data_x) + abs(max(data_x))
    y1 = max(data_y) + abs(max(data_y))
    if constraint_x is not None:
        # plt.plot([constraint_x,constraint_x], [y0, y1],
        #         color='tab:red')
        ax.fill([x0, constraint_x, constraint_x, x0], [y0, y0, y1, y1],
                color='tab:red', alpha=0.3)
    if constraint_y is not None:
        # plt.plot([x0, x1], [constraint_y,constraint_y],
        #         color='tab:red')
        ax.fill([x0, x1, x1, x0], [y0, y0, constraint_y, constraint_y],
                color='tab:red', alpha=0.3)

    # Plot data
    plt.plot(data_x, data_y, linewidth=1, color=color, alpha=0.5)
    plt.scatter(data_x, data_y, color=color, marker='o', s=4)
    print('Max std error (x): %.5f' % max(data_x_std))
    print('Max std error (y): %.5f' % max(data_y_std))
    # ax.errorbar(data_x, data_y,
    #             xerr=data_x_std,
    #             yerr=data_y_std,
    #             color=color, alpha=0.5,
    #             marker='o', markersize=2,
    #             linewidth=1, elinewidth=1, 
    #             capsize=3, capthick=1)

    # Plot Pareto frontier
    # ax.errorbar([data_x[frontier] for frontier in pareto_frontier],
    #             [data_y[frontier] for frontier in pareto_frontier],
    #             xerr=[data_x_std[frontier] for frontier in pareto_frontier],
    #             yerr=[data_y_std[frontier] for frontier in pareto_frontier],
    #             color='tab:blue',
    #             marker='x', markersize=4,
    #             linewidth=0, elinewidth=1, 
    #             capsize=3, capthick=1)
    ax.scatter([data_x[frontier] for frontier in pareto_frontier],
               [data_y[frontier] for frontier in pareto_frontier],
               color='tab:blue',
               marker='x', s=20, zorder=10)

    x_diff = max(data_x) - min(data_x)
    y_diff = max(data_y) - min(data_y)
    x_reldiff = 0.06
    y_reldiff = 0.09
    plt.xlim([min(data_x) - x_diff * x_reldiff, max(data_x) + x_diff * x_reldiff])
    plt.ylim([min(data_y) - y_diff * y_reldiff, max(data_y) + y_diff * y_reldiff])

    if label_x is None:
        plt.xlabel('Virtual neighbors - user %d' % users[0], fontsize=fontsize)
    else:
        plt.xlabel(label_x, fontsize=fontsize)

    if label_y is None:
        plt.ylabel('Virtual neighbors - user %d' % users[0], fontsize=fontsize)
    else:
        plt.ylabel(label_y, fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # Set the axis to have only a few major ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(num_x_ticks))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(num_y_ticks))

    if save:
        if dark == True:
            filename = 'figs/DARK_'
        else:
            filename = 'figs/'
        filename += 'pareto_vneighs-vary_%s-constrs_%s_%s-%s_%s_n%d-p_gen%.3f-q_swap%.3f' \
                    '-p_swap%.3f-p_cons%.3f-cutoff%d-max_links_swapped%d-qbits_per_channel%d' \
                    '-N_samples%d-total_time%d-randomseed%s.pdf' % (varying_param,
                                                                    constraint_x, constraint_y, protocol,
                                                                    topology, n, p_gen, q_swap, p_swap, p_cons, cutoff,
                                                                    max_links_swapped,
                                                                    qbits_per_channel, N_samples, total_time,
                                                                    randomseed)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return pareto_frontier


def plot_avgs_vs_param(users, varying_array, varying_param, protocol, data_type, topology, n, p_gen, q_swap, p_swap,
                       p_cons, cutoff, max_links_swapped, qbits_per_channel, N_samples, total_time, randomseed,
                       dark=False, legend='levels', save=False, steady_state_window=None, steady_state_force_find=False,
                       x_cm=8, y_cm=5, fontsize=8, xlimits=None, ylimits_deg=None, ylimits_neigh=None,
                       num_y_ticks_vdeg=4, num_y_ticks_vneigh=4, optimal_value=True, **kwargs):
    '''Plot average virtual metrics versus some parameter'''

    #####################
    ## LOAD DATA ##
    #####################
    avg_vdegs_sim = [[[] for q in varying_array] for _ in range(n)]
    avg_vneighs_sim = [[[] for q in varying_array] for _ in range(n)]
    std_vdegs_sim = [[[] for q in varying_array] for _ in range(n)]
    std_vneighs_sim = [[[] for q in varying_array] for _ in range(n)]

    # print('WARNING: presence of steady state not checked')

    if protocol == 'ndsrs':
        raise ValueError('Function unavailable for node-dependent protocols')
    elif protocol not in ['srs', 'rprs']:
        raise ValueError('Unknown protocol')

    for idx in tqdmn(range(len(varying_array)), 'q_swap', leave=False):
        if varying_param == 'q_swap':
            q_swap = varying_array[idx]
        else:
            raise ValueError('Varying parameter not implemented.')

        sim_data = load_data_cd(protocol, data_type, topology, n, p_gen, q_swap,
                                p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel,
                                N_samples, total_time, randomseed)
        avg_vdegrees = sim_data['avg_vdegrees']
        avg_vneighs = sim_data['avg_vneighs']
        std_vdegrees = sim_data['std_vdegrees']
        std_vneighs = sim_data['std_vneighs']

        number_of_nodes = len(avg_vdegrees)  # In tree networks, this is different from n

        for node in range(number_of_nodes):
            # Find steady state
            if steady_state_window is None:
                # If no window is specified, we assume the steady state was reached
                vdegs_ss_idx = True
                vneighs_ss_idx = True
            else:
                physical_degrees = kwargs['physical_degrees']

                if steady_state_force_find:  # Ensure that we find a steady state
                    # by making errors progressively large
                    vdegs_ss_idx = None
                    vneighs_ss_idx = None
                    error_increase = 1
                    while vdegs_ss_idx is None:
                        error_vdeg = error_increase * physical_degrees[node] * qbits_per_channel / np.sqrt(N_samples)
                        vdegs_ss_idx = find_steady_state(avg_vdegrees[node], error_vdeg,
                                                         steady_state_window,
                                                         window_type='shrinking')
                        error_increase = error_increase + 0.1
                    error_increase = 1
                    while vneighs_ss_idx is None:
                        error_vneigh = error_increase * min(physical_degrees[node] * qbits_per_channel,
                                                            number_of_nodes) / np.sqrt(N_samples)
                        vneighs_ss_idx = find_steady_state(avg_vneighs[node], error_vneigh,
                                                           steady_state_window,
                                                           window_type='shrinking')
                        error_increase = error_increase + 0.1
                else:
                    error_vdeg = physical_degrees[node] * qbits_per_channel / np.sqrt(N_samples)
                    vdegs_ss_idx = find_steady_state(avg_vdegrees[node], error_vdeg,
                                                     steady_state_window, window_type='shrinking')
                    error_vneigh = min(physical_degrees[node] * qbits_per_channel,
                                       number_of_nodes) / np.sqrt(N_samples)
                    vneighs_ss_idx = find_steady_state(avg_vneighs[node], error_vneigh,
                                                       steady_state_window, window_type='shrinking')

            if vdegs_ss_idx is not None:
                avg_vdegs_sim[node][idx] = avg_vdegrees[node][-1]
                std_vdegs_sim[node][idx] = std_vdegrees[node][-1]
            else:
                avg_vdegs_sim[node][idx] = None
                std_vdegs_sim[node][idx] = None

            if vneighs_ss_idx is not None:
                avg_vneighs_sim[node][idx] = avg_vneighs[node][-1]
                std_vneighs_sim[node][idx] = std_vneighs[node][-1]
            else:
                avg_vneighs_sim[node][idx] = None
                std_vneighs_sim[node][idx] = None

    #####################
    ## PLOTS ##
    #####################

    if varying_param == 'q_swap':
        xlabel = '$q$'
    else:
        raise ValueError('Varying parameter not implemented.')

    cmap = plt.cm.get_cmap('viridis')
    if dark == True:
        plt.style.use('dark_background')
        main_color = 'w'
        if len(users) == 3:
            colors = [cmap(0.3), cmap(0.7), cmap(0.99)]
        else:
            colors = [cmap(i / len(users)) for i in range(len(users))]
    else:
        plt.style.use('default')
        main_color = 'k'
        if len(users) == 3:
            colors = [cmap(0), cmap(0.45), cmap(0.8)]
        else:
            colors = [cmap(i / len(users)) for i in range(len(users))]

    markers = ['o', 'v', 'x', 's', 'd']

    # Virtual degree
    if True:
        fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))
        for idx, node in enumerate(users):
            if legend == 'nodes':
                label = 'Node %d' % node
            elif legend == 'levels':
                label = 'Level %d node' % node
            plt.plot(varying_array, avg_vdegs_sim[node],
                     color=colors[idx],
                     marker=markers[idx], markersize=4,
                     label=label)
            print('Max std error (vdegs): %.5f' % max(2
                                                      * np.array(std_vdegs_sim[node]) / np.sqrt(N_samples)))
            # ax.errorbar(varying_array, avg_vdegs_sim[node],
            #     yerr=2*np.array(std_vdegs_sim[node])/np.sqrt(N_samples),
            #     linestyle='-', color=colors[idx],
            #     marker=markers[idx], markersize=3,
            #     linewidth=1, elinewidth=1,
            #     capsize=3, capthick=1,
            #     label=label)

        if xlimits == None:
            plt.xlim(varying_array[0], varying_array[-1])
        else:
            plt.xlim(xlimits[0], xlimits[1])
        if ylimits_deg == None:
            plt.gca().set_ylim(bottom=0)
        else:
            plt.ylim(ylimits_deg[0], ylimits_deg[1])
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel('$k_i$', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # Set the y-axis to have only three major ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_y_ticks_vdeg))
        if len(users) < 6:
            plt.legend(fontsize=fontsize, loc='upper right', ncol=1)

        if save:
            if dark == True:
                filename = 'figs/DARK_'
            else:
                filename = 'figs/'

            filename += 'avg_vdegrees_vs_%s_%s_%s_n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f' \
                        '-p_cons%.3f-cutoff%d-max_links_swapped%d-qbits_per_channel%d' \
                        '-N_samples%d-total_time%d-randomseed%s.pdf' % (varying_param, protocol,
                                                                        topology, n, p_gen, q_swap, p_swap, p_cons,
                                                                        cutoff, max_links_swapped,
                                                                        qbits_per_channel, N_samples, total_time,
                                                                        randomseed)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # Virtual neighborhood
    if True:
        fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))
        for idx, node in enumerate(users):
            if legend == 'nodes':
                label = 'Node %d' % node
            elif legend == 'levels':
                label = 'Level %d node' % node
            try:
                plt.plot(varying_array, avg_vneighs_sim[node],
                         color=colors[idx],
                         marker=markers[idx], markersize=4,
                         label=label)
                print('Max std error (vneighs): %.5f' % max(2
                                                            * np.array(std_vneighs_sim[node]) / np.sqrt(N_samples)))
                # ax.errorbar(varying_array, avg_vneighs_sim[node],
                #     yerr=2*np.array(std_vneighs_sim[node])/np.sqrt(N_samples),
                #     linestyle='-', color=colors[idx],
                #     marker=markers[idx], markersize=3,
                #     linewidth=1, elinewidth=1,
                #     capsize=3, capthick=1,
                #     label=label)
            except:
                print(std_vneighs_sim[node])
            # Plot line at optimal value
            max_value = max(avg_vneighs_sim[node])
            max_index = avg_vneighs_sim[node].index(max_value)
            plt.plot([varying_array[max_index], varying_array[max_index]],
                     [0, max_value],
                     linestyle=':', color=colors[idx], alpha=0.5)

        if xlimits == None:
            plt.xlim(varying_array[0], varying_array[-1])
        else:
            plt.xlim(xlimits[0], xlimits[1])
        if ylimits_neigh == None:
            plt.gca().set_ylim(bottom=0)
        else:
            plt.ylim(ylimits_neigh[0], ylimits_neigh[1])
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel('$v_i$', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # Set the y-axis to have only three major ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_y_ticks_vneigh))
        if len(users) < 6:
            plt.legend(fontsize=fontsize, loc='upper right', ncol=1)

        if save:
            if dark == True:
                filename = 'figs/DARK_'
            else:
                filename = 'figs/'
            filename += 'avg_vneighs_vs_%s_%s_%s_n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f' \
                        '-p_cons%.3f-cutoff%d-max_links_swapped%d-qbits_per_channel%d' \
                        '-N_samples%d-total_time%d-randomseed%s.pdf' % (varying_param, protocol,
                                                                        topology, n, p_gen, q_swap, p_swap, p_cons,
                                                                        cutoff, max_links_swapped,
                                                                        qbits_per_channel, N_samples, total_time,
                                                                        randomseed)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def plot_colormap(x_array, y_array, data, x_label, y_label, cbar_label, cbar_max=5, cbar_min=0, annotate_min=False,
                  annotate_max=False, annotation_error=0, filename=None):
    '''Generate a 2D colormap. We assume all data takes positive values.
        We assume x_array and y_array are equispaced arrays.
        ---Inputs---
            · ...
            · cbar_max: (int) maximum value of the colorbar.
            · annotate_min:   (bool) if True, annotates the value in the cell
                            corresponding to the smallest values of the varying
                            parameters.
            · annotate_max:   (bool) if True, annotates the value in the cell
                            corresponding to the largest values of the varying
                            parameters.
            · savefig:  (bool) if True, saves figure.'''

    ### FONT STYLE OF NPJ QUANTUM ###
    plt.rcParams["font.family"] = "Arial"
    x_cm = 8
    y_cm = 5
    fontsizes = 8
    fontsizes_ticks = 8

    dx = (x_array[1] - x_array[0]) / 2
    dy = (y_array[1] - y_array[0]) / 2

    fig, ax = plt.subplots(figsize=(x_cm / 2.54, y_cm / 2.54))

    data = np.array(data)
    surfmax = np.max(data)
    surfmin = np.min(data)
    cbar_max = max(cbar_max, surfmax)
    cbar_min = min(cbar_min, surfmin)
    cbar_mid = cbar_min + (cbar_max - cbar_min) / 2

    cmap = plt.cm.get_cmap('Blues')
    cont = ax.imshow(np.flip(data.T, 0), cmap=cmap,
                     extent=[x_array[0] - dx, x_array[-1] + dx,
                             y_array[0] - dy, y_array[-1] + dy],
                     vmin=cbar_min, vmax=cbar_max)
    ax.set_aspect(aspect="auto")

    # Ticks and labels #

    ax.set_xticks(x_array[::2])
    x_minor_intervals = 2  # Number of minor intervals between two major ticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(x_minor_intervals))

    ax.set_yticks(y_array[::2])
    y_minor_intervals = 2  # Number of minor intervals between two major ticks
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(y_minor_intervals))

    plt.xlabel(x_label, fontsize=fontsizes)
    plt.ylabel(y_label, fontsize=fontsizes)
    ax.tick_params(labelsize=fontsizes_ticks)

    # Colorbar #

    fontsize_cbar_label = fontsizes
    cbar = fig.colorbar(cont, ax=ax, aspect=10)
    cbar.set_label(cbar_label, fontsize=fontsize_cbar_label)
    if surfmax == surfmin:
        cbar.set_ticks([0, surfmax])
        cbar.ax.set_yticklabels(['${:.2f}$'.format(0),
                                 '${:.2f}$'.format(surfmax)])
    else:
        cbar.set_ticks([cbar_min, cbar_mid, cbar_max])
        cbar.ax.set_yticklabels(['${:.2f}$'.format(cbar_min),
                                 '${:.2f}$'.format(cbar_mid),
                                 '${:.2f}$'.format(cbar_max)])
    cbar.ax.tick_params(labelsize=fontsizes_ticks)

    # Annotated colormap #

    for iix, ii in enumerate(x_array):
        for jjx, jj in enumerate(y_array):
            if ((annotate_min and data[iix][jjx] - annotation_error <= surfmin) or
                    (annotate_max and data[iix][jjx] + annotation_error >= surfmax)):
                if data.T[jjx, iix] > cbar_min + (cbar_max - cbar_min) / 2:
                    textcolor = 'w'
                else:
                    print(ii, jj)
                    textcolor = 'k'
                if data.T[jjx, iix] > 10:
                    annotation = '$%.1f$' % data.T[jjx, iix]
                else:
                    annotation = '$%.1f$' % data.T[jjx, iix]
                text = ax.text(ii, jj, annotation, fontsize=fontsizes_ticks,
                               ha='center', va='center', color=textcolor)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_entangled_links(S, cutoff=np.inf, layout='random', show_link_data=False, arc_radius=0.5):
    '''S is the qubit registers.
        ---Inputs---
            · cutoff:   (int) used to set the transparency of each link.
            · layout:   (str) 'chain', 'squared', 'random'.'''

    # Find n (number of nodes) and r (number of qubits per neighbor per node)
    n = len(S)
    for i in range(n):
        for j in range(n):
            if not S[i][j] == 0:
                r = len(S[i][j])
                break

    # Node distribution
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))  # Include disconnected nodes

    if layout == 'chain':
        pos = {}
        for i in range(n):
            pos[i] = [i / (n - 1), 0]
    elif layout == 'squared':
        l = int(n ** 0.5)
        pos = {}
        for i in range(n):
            if i < l:
                x = 0
            else:
                x = int((i - (l - 1) - 1e-10) // l + 1)
            y = i % l
            pos[i] = [x / (l - 1), y / (l - 1)]
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    else:
        raise ValueError('Unknown layout')

    # Draw links
    rad_extra = []
    l = 1
    while l < np.ceil(r / 2) + 1:
        rad_extra += [l, -l]
        l += 1
    ax = plt.gca()

    # Generate network data
    id_set = set()
    for i in range(n):
        for j in range(n):
            if not S[i][j] == 0:
                for m, qubit in enumerate(S[i][j]):
                    if not qubit is None:
                        qubit1_id = (i, j, m)
                        qubit2_id = tuple(qubit[2])
                        if (qubit1_id not in id_set) and (qubit2_id not in id_set):
                            id_set.add(qubit1_id)
                            id_set.add(qubit2_id)
                            if show_link_data:
                                print(qubit[0], qubit1_id, qubit2_id)
                            i2 = qubit2_id[0]
                            G.add_edge(i, i2)

                            # Plot edge
                            if layout == 'chain':
                                rad_offset = (np.abs(i2 - i) - 1) * arc_radius
                                rad = rad_offset * np.sign(rad_extra[m]) + rad_extra[m] * arc_radius / r
                                connect_style = 'arc3,rad=%s' % str(rad)
                            elif layout == 'squared':
                                connect_style = 'arc3,rad=%s' % str(0.3 * m + 0.1)
                            elif layout == 'random':
                                connect_style = 'arc3,rad=%s' % str(0.3 * m)
                            else:
                                connect_style = 'arc3,rad=%s' % str(0.3 * m)
                            ax.annotate('', xy=pos[i], xycoords='data',
                                        xytext=pos[i2], textcoords='data',
                                        arrowprops=dict(arrowstyle='-', color='k',
                                                        linewidth=2,
                                                        alpha=1 - S[i][j][m][0] / (cutoff + 0.1),
                                                        connectionstyle=connect_style))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=100, alpha=1)

    plt.axis('off')
    plt.show()


# ---------------------------------------------------------------------------
# ----------------------------- DATA STORAGE --------------------------------
# ---------------------------------------------------------------------------
def check_data_cd(protocol, data_type, topology, n, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                  qbits_per_channel, N_samples, total_time, randomseed, nd_label=None):
    '''If simulation_rprs() has been run and saved for this set of
        parameters, return True. Otherwise, return False.
        ---Inputs---
            · nd_label: (str) label used to identify data from protocols with node-dependent
                        parameters. Otherwise we would need to save all the parameters in
                        the file name.'''
    if protocol not in ['srs', 'rprs', 'ndsrs']:
        raise ValueError('Unknown protocol')
    elif protocol == 'ndsrs':
        filename = 'data-%s/%s/%s/n%d-%s-p_gen%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, nd_label,
                                                   p_gen, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    else:
        filename = 'data-%s/%s/%s/n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, p_gen,
                                                   q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    if Path(filename).exists():
        return True
    else:
        return False


def save_data_cd(data, protocol, data_type, topology, n, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                 qbits_per_channel, N_samples, total_time, randomseed, nd_label=None):
    '''Save data for this set of parameters.
        ---Inputs---
            · ... '''
    # Create data directory if needed
    try:
        os.mkdir('data-%s' % protocol)
    except FileExistsError:
        pass
    try:
        os.mkdir('data-%s/%s/' % (protocol, data_type))
    except FileExistsError:
        pass
    try:
        os.mkdir('data-%s/%s/%s/' % (protocol, data_type, topology))
    except FileExistsError:
        pass

    # Save data
    if protocol == 'ndsrs':
        filename = 'data-%s/%s/%s/n%d-%s-p_gen%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, nd_label,
                                                   p_gen, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    else:
        filename = 'data-%s/%s/%s/n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, p_gen,
                                                   q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    if data_type == 'all':
        _data = {'vdegrees': data[0], 'vneighs': data[1]}
    elif data_type == 'avg':
        _data = {'avg_vdegrees': data[0], 'avg_vneighs': data[1],
                 'std_vdegrees': data[2], 'std_vneighs': data[3]}
    else:
        raise ValueError('Unknown data_type')
    if protocol == 'ndsrs':
        _data['q_swap_vec'] = q_swap
    with open(filename, 'wb') as handle:
        pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_cd(protocol, data_type, topology, n, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                 qbits_per_channel, N_samples, total_time, randomseed, nd_label=None):
    '''Load data obtained via simulation_rprs().
        ---Inputs---
            · ...
        ---Outputs---
            · ... '''
    if protocol == 'ndsrs':
        filename = 'data-%s/%s/%s/n%d-%s-p_gen%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, nd_label,
                                                   p_gen, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    else:
        filename = 'data-%s/%s/%s/n%d-p_gen%.3f-q_swap%.3f-p_swap%.3f-p_cons%.3f' \
                   '-cutoff%d-max_links_swapped%d-qbits_per_channel%d-N_samples%d' \
                   '-total_time%d-randomseed%s' % (protocol, data_type, topology, n, p_gen,
                                                   q_swap, p_swap, p_cons, cutoff, max_links_swapped,
                                                   qbits_per_channel, N_samples, total_time, randomseed)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data
