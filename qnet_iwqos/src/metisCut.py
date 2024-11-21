import networkx as nx
import random

import numpy as np
import pymetis


def partGraphInit():
    # 创建一个带权的随机图
    G = nx.Graph()
    # 添加10个节点
    G.add_nodes_from(range(10))
    # 随机添加边和权重
    for i in range(10):
        for j in range(i + 1, 10):
            if random.random() < 0.5:  # 以一定概率添加边
                G.add_edge(i, j, weight=random.randint(1, 10))

    # 转换图为pymetis的输入格式
    adjacency_list = [list(G.neighbors(i)) for i in range(10)]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 使用pymetis的part_graph_kway方法划分图
    num_parts = 3  # 划分成3个子图
    cuts, membership = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    # 打印节点的分组
    print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in range(num_parts)]

    # 打印每个子图的节点
    for i, sg in enumerate(subgraphs, start=1):
        print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum

def weight4metis(cx_weight):
    xadj = [0]
    now_index = 0
    adjncy = []
    eweight = []
    for i in range(len(cx_weight)):
        for j in range(len(cx_weight)):
            # print(cx_weight[i][j])
            if cx_weight[i][j]:
                adjncy.append(j)
                eweight.append(cx_weight[i][j])
                now_index = now_index + 1
        xadj.append(now_index)
    return xadj, adjncy, eweight

def metis_zmz(G, k, randomseed):
    random.seed(randomseed)
    np.random.seed(randomseed)
    m = nx.adjacency_matrix(G)
    # print(m)
    # print(nx.adjacency_matrix(G).todense())
    num_parts = k
    xadj, adjncy, eweight = weight4metis(nx.adjacency_matrix(G).todense().tolist())
    cuts, membership = pymetis.part_graph(nparts=num_parts, adjncy=adjncy, xadj=xadj, eweights=eweight)
    # print(cuts)
    # print(membership)

    # 打印节点的分组
    # print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in
                 range(num_parts)]

    # 打印每个子图的节点
    # for i, sg in enumerate(subgraphs, start=1):
    #     print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    # print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum

def partGraph(G, k):
    # G: 量子线路生成的带权图 k: 划分成的子图数目

    # 转换图为pymetis的输入格式
    adjacency_list = [list(G.neighbors(i)) for i in range(len(G.nodes))]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 使用pymetis的part_graph方法划分图
    num_parts = k  # 划分成3个子图
    cuts, membership = pymetis.part_graph(num_parts, eweights=weights, adjacency=adjacency_list)
    # cuts, membership = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    # 打印节点的分组
    print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in range(num_parts)]

    # 打印每个子图的节点
    for i, sg in enumerate(subgraphs, start=1):
        print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum
#
# def partGraphKway():
#     import networkx as nx
#     import pymetis
#     import random
#
#     # 创建一个带权重的随机图
#     G = nx.gnm_random_graph(10, 20)
#     for (u, v) in G.edges():
#         G.edges[u, v]['weight'] = random.randint(1, 10)
#
#     # 将NetworkX图转换为Pymetis的输入格式
#     adjacency_list = [list(G.neighbors(i)) for i in range(len(G))]
#     # 为Pymetis准备边权重
#     edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
#
#     # 调用pymetis分区函数，将图划分为3个子图
#     cuts, membership = pymetis.part_graph_kway(3, adjacency=adjacency_list, eweights=edge_weights)
#
#     # 输出每个节点的子图分配
#     print("Node Subgraph Membership:", membership)
#
#     # 计算被分割的边的权重总和
#     cut_weight_sum = 0
#     for u, v in G.edges():
#         if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
#             cut_weight_sum += G[u][v]['weight']
#
#     print("Total Weight of Cut Edges:", cut_weight_sum)


if __name__ == "__main__":
    metis_zmz()