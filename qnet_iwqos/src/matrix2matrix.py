import numpy as np
from scipy.optimize import linear_sum_assignment


def pad_matrix(matrix, target_shape):
    padded_matrix = np.zeros(target_shape)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

def cal_matrix(mat_A):
    list_A = [0]*len(mat_A)
    for i in range(len(mat_A)):
        for j in range(len(mat_A)):
            list_A[i] = list_A[i]+mat_A[i][j]
    return list_A

def map_nodes(cost_matrix):
    # adjacency_A = np.array(graph_A)
    # adjacency_B = np.array(graph_B)
    # new_mat = cost_matrix(graph_A, graph_B)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def cost_matrix(graph_A, graph_B):
    max_nodes = max(len(graph_A), len(graph_B))
    padded_graph_A = pad_matrix(np.array(graph_A), (max_nodes, max_nodes))

    A_list = cal_matrix(padded_graph_A)
    B_list = cal_matrix(graph_B)
    new_mat = []
    for i in range(len(padded_graph_A)):
        a = []
        for j in range(len(padded_graph_A)):
            # if A_list[i] = 0
            # a.append(abs((A_list[i] - B_list[j])))
            if A_list[i] - B_list[j] > 0:
                a.append(A_list[i] - B_list[j])
            else:
                a.append(0)
        new_mat.append(a)
    return new_mat


def map_nodes_greedy(graph_A, graph_B):
    # adjacency_A = np.array(graph_A)
    # adjacency_B = np.array(graph_B)
    max_nodes = max(len(graph_A), len(graph_B))
    padded_graph_A = pad_matrix(np.array(graph_A), (max_nodes, max_nodes))

    A_list = cal_matrix(padded_graph_A)
    B_list = cal_matrix(graph_B)
    b_sorted_indices = sorted(range(len(B_list)), key=lambda i: B_list[i], reverse=True)

    # 按照 A 从大到小的顺序获取对应索引
    a_sorted_indices = sorted(range(len(A_list)), key=lambda i: A_list[i], reverse=True)

    # 创建一个空的映射列表
    mapping = [None] * len(B_list)

    # 将 B 中的值映射到 A 的相应位置
    for idx in range(len(a_sorted_indices)):
        mapping[a_sorted_indices[idx]] = b_sorted_indices[idx]

    circuit = [i for i in range(len(mapping))]

    return circuit, mapping


def cost_matrix_sliced(normalize_sliced_subcircuits_communication, graph_B):
    # sliced_subcircuits_communication_list = normalize_sliced_subcircuits_communication.sum(axis = 2)
    num_devices = len(graph_B)
    cost_all = np.zeros((num_devices, num_devices))
    for slice in normalize_sliced_subcircuits_communication:
        slice_cost = cost_matrix(slice, graph_B)
        cost_all += slice_cost
    return cost_all

def compute_mapping_information(cost_matrix, subcircuits_allocation):
    # adjacency_A = np.array(graph_A)
    # adjacency_B = np.array(graph_B)

    mapping_cost_information = {}
    # row_ind, col_ind = linear_sum_assignment(new_mat)
    cost = 0
    # if row_ind == None and col_ind == None:
    #     for i  in range(len(new_mat)):
    #         cost += new_mat[i][i]
    #     mapping_cost_information['cost'] = cost
    # else:
    for i, j in subcircuits_allocation.items():
        cost += cost_matrix[i][j]
    zero_count = 0
    for row  in cost_matrix:
        for num in row:
            if num == 0:
                zero_count += 1
    mapping_cost_information['zero_num'] = zero_count
    mapping_cost_information['cost'] = cost
    return mapping_cost_information


# graph_A = [[0, 1, 0],
#            [1, 0, 1],
#            [0, 1, 0]]
#
# graph_B = [[0, 0, 1, 0],
#            [0, 0, 1, 0],
#            [1, 1, 0, 1],
#            [0, 0, 1, 0]]
#
# max_nodes = max(len(graph_A), len(graph_B))
# padded_graph_A = pad_matrix(np.array(graph_A), (max_nodes, max_nodes))
#
# row_ind, col_ind = map_nodes(padded_graph_A, graph_B)
#
# print("result")
# for i, j in zip(row_ind, col_ind):
#     print(f"nodeA {i} to nodeB {j}")