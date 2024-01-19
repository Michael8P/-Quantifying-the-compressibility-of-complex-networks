import scipy.io
import scipy
import numpy as np
def avg_clustering_coefficient(G):
    if issparse(G):
        G = G.toarray()
    else:
        G = G
    # Check if the input matrix is int32
    if G.dtype != np.int32:
        G = G.astype(np.int32)

    # Function to check if matrix is symmetric
    def is_symmetric(matrix):
        return np.allclose(matrix, matrix.T)

    # Validate input
    if not is_symmetric(G):
        raise ValueError("Input matrix must be symmetric")

    # Make sure graph is unweighted
    G[G != 0] = 1
    
    deg = np.sum(G, axis=1) # Determine node degrees 
    cn =  np.diag(np.dot(np.dot(G, np.triu(G)), G)) # Number of Triangles for each node
    
    
    # The local clustering coefficient of each node
    c = np.zeros(len(deg))
    mask = deg > 1
    c[mask] = 2 * cn[mask] / (deg[mask] * (deg[mask] - 1))
    
    acc = np.mean(c[mask])

    return acc, c
