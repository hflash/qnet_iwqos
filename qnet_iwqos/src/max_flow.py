import numpy as np
from collections import deque


def bfs(residual_graph, parent, source, sink):
    visited = [False] * len(residual_graph)
    queue = deque()
    queue.append(source)
    visited[source] = True

    while queue:
        u = queue.popleft()
        for ind, val in enumerate(residual_graph[u]):
            if not visited[ind] and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
                if ind == sink:
                    return True
    return False


def edmonds_karp(graph, source, sink):
    residual_graph = np.copy(graph)
    parent = [-1] * len(graph)
    max_flow = 0

    while bfs(residual_graph, parent, source, sink):
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]
        max_flow += path_flow

        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]

    return max_flow, residual_graph


def extract_paths(residual_graph, source, sink):
    paths = []

    def dfs(u, path):
        if u == sink:
            paths.append(path.copy())
            return
        for v, cap in enumerate(residual_graph[u]):
            if cap > 0 and v not in path:
                path.append(v)
                dfs(v, path)
                path.pop()

    dfs(source, [source])
    return paths


def solve_disjoint_paths(adj_matrix, node_pairs):
    num_nodes = len(adj_matrix)
    source, sink = num_nodes, num_nodes + 1
    graph = np.zeros((num_nodes + 2, num_nodes + 2))
    graph[:num_nodes, :num_nodes] = adj_matrix

    for s, _ in node_pairs:
        graph[source, s] = np.inf
    for _, t in node_pairs:
        graph[t, sink] = np.inf

    max_flow, residual_graph = edmonds_karp(graph, source, sink)
    if max_flow >= len(node_pairs):
        paths = extract_paths(residual_graph[:num_nodes, :num_nodes], source, sink)
        return True, paths
    else:
        return False, []


# Example usage
adj_matrix = np.array([
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
])
node_pairs = [(0, 5), (1, 3)]

has_paths, paths = solve_disjoint_paths(adj_matrix, node_pairs)
print("Has Disjoint Paths:", has_paths)
if has_paths:
    print("Paths:", paths)
