#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
from random import sample
from numpy import inf, size
import itertools
import matplotlib.pyplot as plt
import random
np.seterr(divide = 'ignore') 
G = pd.read_csv("ZACHARY_Karate_symmetric_binary.csv", header=None).values
heuristic=1
num_pairs=2
def rate_distortion(G, heuristic, num_pairs):
    # Size of network
    N = len(G)
    E = int(np.sum(G[:])/2)
    
    # Variables to save
    S = np.zeros((1,N), dtype=float) #Upper bound on entropy rate after clustering
    S_low = np.zeros((1,N), dtype=float)#Lower bound on entropy rate
    clusters = [[] for _ in range(N)] #clusters{n} lists the nodes in each of the n clusters
    Gs = [[] for _ in range(N)] #Gs{n} is the joint transition probability matrix for n clusters
    
    # Transition proability matrix
    P_old = np.divide(G, np.transpose(np.tile(sum(G), (N,1))))   
    
    
    
    # Compute steady-state distribution
    D, p_ss = sla.eigs(np.transpose(P_old),return_eigenvectors=True)  #works for all networks 
    D = D.real
    p_ss = p_ss.real
    p_ss = -1 * p_ss
    ind = int(np.round(np.max(D))) - 1 
    # p_ss = p_ss[:,ind]/sum(p_ss[:,ind]) # This is only for directed networks
    p_ss = np.sum(G,axis = 1) / np.sum(G)  # Only true for undirected networks
    
    p_ss_old = p_ss
    
    # Calculate initial entropy:
    logP_old = np.log2(P_old)
    logP_old[logP_old == -inf] = 0   
    S_old = -np.sum(np.multiply(p_ss_old, np.sum(np.multiply(P_old, logP_old), axis=1)))   
    P_joint = np.multiply(P_old, np.transpose(np.tile(p_ss_old, (N,1)))) 
    P_low = P_old
    # Record inital values
    S[:,-1] = S_old
    S_low[:,-1] = S_old
    clusters[-1] = [x for x in range(0, N)] 
    Gs[-1] = G  

    # Loop over the number of clusterings
    for i in range(N-1,1,-1):
    #different sets of node pairs to try:
        if heuristic == 1:
        #try combining all pairs:
            pairs = np.array(list(itertools.combinations([x for x in range(i+1)], 2)))
            I = pairs[:,0]
            J = pairs[:,1]
        elif heuristic == 2:
            #pick num_pair node pairs at random
            pairs = np.array(list(itertools.combinations([x for x in range(i+1)], 2)))
            inds = sample(range(size(pairs,1)), np.min([num_pairs, size(pairs,1)])); 
            I = pairs[inds,0]
            J = pairs[inds,1]
        elif heuristic == 3:
            # Try combining all pairs connected by an edge
            I, J = np.where(np.triu(P_old + np.transpose(P_old), k=1))
        elif heuristic == 4:
            # pick num_paid nodes at random that are connected by an edge
            I, J = np.where(np.triu(P_old + np.transpose(P_old), k=1))
            pair_inds = sample(range(len(I)), min([num_pairs, len(I)]))
            I = I[pair_inds]
            J = J[pair_inds]
        elif heuristic == 5:
            # Pick num_pairs node pairs with largest joint transition probabilities
            P_joint_symm = np.triu(P_joint + np.transpose(P_joint), k=1)
            matrix = P_joint_symm.ravel()
            K = min([num_pairs, np.sum(P_joint_symm.ravel() > 0)])
            indices = np.argpartition(matrix, -K)[-K:]
            indices = indices[np.argsort(matrix[indices])][::-1]
            subscripts = np.unravel_index(indices, (i+1, i+1))
            I = subscripts[0]
            J = subscripts[1]
        elif heuristic == 6:
            # pick num_pair node pairs with largest joint transition probabilities plus self-transition probabilities
            P_joint_symm = np.triu(P_joint + np.transpose(P_joint) + np.transpose(np.tile(np.diag(P_joint), (i+1,1))) + np.transpose(np.tile(np.transpose(np.diag(P_joint)), (i+1,1))) ,k=1)
            matrix = P_joint_symm.ravel()
            K = min([num_pairs, np.sum(P_joint_symm.ravel() > 0)])
            indices = np.argpartition(matrix, -K)[-K:]
            indices = indices[np.argsort(matrix[indices])][::-1]
            subscripts = np.unravel_index(indices, (i+1, i+1))
            I = subscripts[0]
            J = subscripts[1]     
        elif heuristic == 7:
            # Pick num_pairs node pairs with largest combined stationary probabilities
            P_ss_temp = np.triu(np.transpose(np.tile(p_ss_old, (i+1,1))) + np.tile(np.transpose(p_ss_old), (i+1,1)), k=1) 
            matrix = P_ss_temp.ravel()
            K = min([num_pairs,len(list(itertools.combinations(range(i+1), 2)))])
            indices = np.argpartition(matrix, -K)[-K:]
            indices = indices[np.argsort(matrix[indices])][::-1]
            subscripts = np.unravel_index(indices, (i+1, i+1))
            I = subscripts[0]
            J = subscripts[1]         
        else:
            print('Variable "setting" is not properly defined.')
    
        #number of pairs 
        num_pairs_temp = len(I)
    
        #keep track of all entropies
        S_all = np.zeros((1, num_pairs_temp), dtype=float)
    
        #loop over the pairs of nodes:
        for ind in range(num_pairs_temp):
            ii = I[ind]
            jj = J[ind]
            inds_not_ij = list(range(0,(ii))) + list(range((ii+1),(jj))) + list(range((jj+1),(i+1)))
            # inds_not_ij = [x - 1 for x in inds_not_ij] issue if 0 then it becomes -1
            #compute new stationary distribution:           
            p_ss_temp = np.append(p_ss_old[inds_not_ij], p_ss_old[ii] + p_ss_old[jj])
            
            #stopped here
            # Compute new transition probabilities:
            P_temp_1 = np.sum(np.multiply(np.transpose(np.tile(p_ss_old[inds_not_ij], (2,1))), P_old[:, [ii,jj]][inds_not_ij]), axis = 1)
            P_temp_1 = P_temp_1 / p_ss_temp[0:-1]
            P_temp_2 = np.sum(np.multiply(np.transpose(np.tile(np.r_[p_ss_old[ii],p_ss_old[jj]] , (i-1,1))), P_old[[ii,jj],:][:,inds_not_ij]),axis=0, keepdims=True)
            P_temp_2 = P_temp_2 / p_ss_temp[-1]
            P_temp_3 = np.sum(np.multiply(np.transpose(np.tile(np.r_[p_ss_old[ii],p_ss_old[jj]], (2,1))), P_old[:,[ii,jj]][[ii,jj], :]))
            P_temp_3 = P_temp_3 / p_ss_temp[-1]
            
            logP_temp_1 = np.log2(P_temp_1)
            logP_temp_1[logP_temp_1 == -inf] = 0
            logP_temp_2 = np.log2(P_temp_2)
            logP_temp_2[logP_temp_2 == -inf] = 0
            logP_temp_3 = np.log2(P_temp_3)
            logP_temp_3 = np.array([logP_temp_3])
            logP_temp_3[logP_temp_3 == -inf] = 0
    
            #Compute change in upper bound on mutual information
            d1 = -sum(np.multiply(np.multiply(p_ss_temp[0:-1], P_temp_1), logP_temp_1))
            d2 = - p_ss_temp[-1]*np.sum(np.multiply(P_temp_2,logP_temp_2))
            d3 = - p_ss_temp[-1]*P_temp_3*logP_temp_3
            d3 = d3[0]
            d4 = np.sum(np.multiply(p_ss_old, np.multiply(P_old[:,ii], logP_old[:, ii])))
            d5 = np.sum(np.multiply(np.multiply(p_ss_old, P_old[:,jj]), logP_old[:,jj]))
            d6 = p_ss_old[ii]*np.sum(np.multiply(P_old[ii,:],logP_old[ii,:]))
            d7 = p_ss_old[jj]*np.sum(np.multiply(P_old[jj,:], logP_old[jj,:]))
            d8 = - p_ss_old[ii]*(P_old[ii,ii]*logP_old[ii,ii] + P_old[ii,jj]*logP_old[ii,jj])
            d9 = - p_ss_old[jj]*(P_old[jj,jj]*logP_old[jj,jj] + P_old[jj,ii]*logP_old[jj,ii])
            dS = d1+d2+d3+d4+d5+d6+d7+d8+d9
            S_temp = (S_old + dS)
            
        
               # Keep track of all entropies:
            S_all[:,ind] = S_temp
    
    
    
         # Find minimum entropy:
        [dummy, min_inds] = np.nonzero(S_all == np.min(S_all))
        temp_mininds = list(min_inds)
        min_ind = np.random.choice(sorted(temp_mininds))
    
        # Save mutual information:
        S_old = S_all[:,min_ind]
        S[:,i-1] = S_old
    
        # Compute old transition probabilities:
        ii_new = int(I[min_ind])
        jj_new = int(J[min_ind])
        
        inds_not_ij = list(range(0,(ii_new))) + list(range((ii_new+1),(jj_new))) + list(range((jj_new+1),(i+1))) 
    
        p_ss_new = np.append(p_ss_old[inds_not_ij], p_ss_old[ii_new] + p_ss_old[jj_new])
        
        
        #P_joint changes
        P_joint = np.multiply(np.transpose(np.tile(p_ss_old, (i+1, 1))), P_old) #this is good
        P_joint = np.c_[np.r_[P_joint[:,inds_not_ij][inds_not_ij,:],np.sum(P_joint[[ii_new,jj_new],:][:,inds_not_ij], axis=0, keepdims=True)], 
                         np.append(np.sum((P_joint[:, [ii_new,jj_new]][inds_not_ij]), axis=1),np.sum(P_joint[np.ix_([ii_new,jj_new],[ii_new,jj_new])]))]        #why did this change
        
        
        
        
        P_old = np.divide(P_joint, np.transpose(np.tile(p_ss_new, (i, 1))))
        p_ss_old = p_ss_new
    
        logP_old = np.log2(P_old)
        logP_old[logP_old == -inf] = 0
    
        # Record clusters and graph:
        
        cluster1 = clusters[i][0:ii_new]
        if (ii_new + 1) > (jj_new - 1):
            cluster2 = []
        else:
            cluster2 = clusters[i][(ii_new + 1):(jj_new)]
        if (jj_new + 1) >= (i + 1):
            cluster3 = []
        else:
            cluster3 = clusters[i][(jj_new + 1): (i + 1)]
            
        cluster4_ii = clusters[i][ii_new]
        cluster4_ii = [cluster4_ii] if isinstance(cluster4_ii,int) else cluster4_ii
        cluster4_jj = clusters[i][jj_new]
        cluster4_jj = [cluster4_jj] if isinstance(cluster4_jj,int) else cluster4_jj
        cluster4 = cluster4_ii + cluster4_jj
        if len(cluster2) == 0 and len(cluster3) == 0:
            cluster1.append(cluster4)
        if len(cluster2) > 0 and len(cluster3) > 0:
            cluster1 = cluster1+cluster2+cluster3
            cluster1.append(cluster4)
        elif len(cluster2) > 0:
            cluster1 = cluster1+cluster2
            cluster1.append(cluster4)
        elif len(cluster3) > 0:
            cluster1 = cluster1+cluster3
            cluster1.append(cluster4)
        clusters[i-1] = cluster1
        Gs[i-1] = P_joint*2*E
    
        # Compute lower bound on mutual information:
        P_low = np.c_[P_low[:,list(range(0,ii_new))+list(range(ii_new+1, jj_new))+list(range(jj_new+1, i+1))] , P_low[:,ii_new] + P_low[:,jj_new]]
    
        logP_low = np.log2(P_low)
        logP_low[logP_low == -inf] = 0
        S_low[:,i-1] = -sum(np.multiply(p_ss , np.sum(np.multiply(P_low, logP_low), axis=1)))

    return(S, S_low, clusters, Gs)
