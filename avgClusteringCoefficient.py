import numpy as np

# assume the graph is represented as a numpy array
graph[graph != 0] = 1

deg = np.sum(graph, axis=1) # determine node degrees
cn = np.diag(graph @ np.triu(graph) @ graph) # number of triangles for each node

# the local clustering coefficient of each node
c = np.zeros_like(deg)
c[deg > 1] = 2 * cn[deg > 1] / (deg[deg > 1] * (deg[deg > 1] - 1))

# average clustering coefficient of the graph
acc = np.mean(c[deg > 1])
