import scipy
import numpy as np
import pandas as pd
def Deg_Heterogeneity(G):
    if issparse(G):
        G = G.toarray()
    else:
        G = G
    if G.dtype != np.int32:
        G = G.astype(np.int32)
    ks = np.sum(G, axis=0)
    N = np.shape(G[0])[0]    
    deg_het_temp = np.sum(np.abs(ks - ks[:, np.newaxis])) / (N * (N - 1) * np.mean(ks))
    return deg_het_temp
