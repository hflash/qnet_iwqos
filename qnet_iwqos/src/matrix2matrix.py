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

def map_nodes(graph_A, graph_B):
    # adjacency_A = np.array(graph_A)
    # adjacency_B = np.array(graph_B)
    max_nodes = max(len(graph_A), len(graph_B))
    padded_graph_A = pad_matrix(np.array(graph_A), (max_nodes, max_nodes))

    A_list = cal_matrix(padded_graph_A)
    B_list = cal_matrix(graph_B)
    new_mat = []
    for i in range(len(padded_graph_A)):
        a=[]
        for j in range(len(padded_graph_A)):
            a.append(abs(A_list[i]-B_list[j]))
        new_mat.append(a)
    row_ind, col_ind = linear_sum_assignment(new_mat)


    return row_ind, col_ind



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