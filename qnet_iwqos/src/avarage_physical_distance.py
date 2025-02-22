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
# @Time     : 2025/1/6 10:15
# @Author   : HFLASH @ LINKE
# @File     : avarage_physical_distance.py
# @Software : PyCharm

import numpy as np


physical_avg_dist_grid = {}
physical_avg_dist_grid["3"]= 2
physical_avg_dist_grid['2']= 1.333

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

def floyd_warshall(adj_matrix):
    """使用 Floyd-Warshall 算法计算最短路径矩阵"""
    num_nodes = adj_matrix.shape[0]
    dist = adj_matrix.copy()

    # 将没有直接连接的节点标记为无穷大
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)  # 对角线为零

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

def average_distance(adj_matrix):
    """计算图中的平均距离"""
    dist_matrix = floyd_warshall(adj_matrix)
    # 忽略无穷大和零（自身），计算有效路径的平均值
    finite_distances = dist_matrix[~np.isinf(dist_matrix) & (dist_matrix != 0)]

    if len(finite_distances) == 0:
        return float('inf')  # 如果没有有效距离，返回无穷大



    return np.mean(finite_distances)

# A = adjacency_squared_hard(3)
# print(A)
# print(floyd_warshall(A))
# print(average_distance(A))
if __name__ == '__main__':
    A = [[0.0, 4, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 0.0, 5, 0.0, 4, 0.0, 0.0, 0.0, 0.0],
         [0.0, 5, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0], [1, 0.0, 0.0, 0.0, 1, 0.0, 1, 0.0, 0.0],
         [0.0, 4, 0.0, 1, 0.0, 5, 0.0, 4, 0.0], [0.0, 0.0, 1, 0.0, 5, 0.0, 0.0, 0.0, 1],
         [0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 4, 0.0], [0.0, 0.0, 0.0, 0.0, 4, 0.0, 4, 0.0, 1],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 1, 0.0]]

    B = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] != 0:
                B[i][j] = 1
            else:
                B[i][j] = 0
    print(floyd_warshall(B))
    print(average_distance(B))
