#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
from random import sample
from numpy import inf
import itertools
import matplotlib.pyplot as plt
from numpy import size


np.seterr(divide = 'ignore') 
G = pd.read_csv("ZACHARY_Karate_symmetric_binary.csv", header=None).values
heuristic=1
num_pairs=2
def rate_distortion(G, heuristic, num_pairs):
    #size of network
    N = len(G)
    E = int(np.sum(G[:])/2)

    #variables to save
    S = np.zeros((1,N), dtype=float) #Upper bound on entropy rate after clustering
    S_low = np.zeros((1,N), dtype=float)#Lower bound on entropy rate
    clusters = [[] for _ in range(N)] #clusters{n} lists the nodes in each of the n clusters
    Gs = [[] for _ in range(N)] #Gs{n} is the joint transition probability matrix for n clusters

    #transition proability matrix
    P_old = np.divide(G, np.transpose(np.tile(sum(G), (N,1))))

    #Compute steady-state distribution
    D, p_ss = sla.eigs(np.transpose(P_old),return_eigenvectors=True)  #works for all networks     #need to rewrite this section
    D = np.round(D.real, decimals=4)
    p_ss = p_ss.real
    ind = int(np.round(np.max(D))) - 1 
    p_ss = np.round(p_ss[:,ind]/sum(p_ss[:,ind]), decimals=4)

    #p_ss = sum(G,2)/sum(G(:)); % Only true for undirected networks
    p_ss_old = p_ss

    #calculate initial entropy:
    logP_old = np.log2(P_old)
    logP_old[logP_old == -inf] = 0
    S_old = np.round(-np.sum(np.multiply(p_ss_old, np.sum(np.multiply(P_old, logP_old), axis=1))), decimals = 4)
    P_joint = np.round(np.multiply(P_old, np.transpose(np.tile(p_ss_old, (N,1)))), decimals=4)
    P_low = P_old
    #Record inital values
    S[:,-1] = S_old
    S_low[:,-1] = S_old
    clusters[-1] = np.transpose(np.array([[x] for x in range(0, N)]))
    Gs[-1] = G

    #loop over the number of clusterings
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
            inds = sample(size(pairs,1), np.min([num_pairs, size(pairs,1)])); 
            I = pairs(inds,1)
            J = pairs(inds,2)
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

            #compute new stationary distribution:           
            p_ss_temp = np.append(p_ss_old[inds_not_ij], p_ss_old[ii] + p_ss_old[jj])
            
            #stopped here
            # Compute new transition probabilities:
            P_temp_1 = np.sum(np.multiply(np.transpose(np.tile(p_ss_old[inds_not_ij], (2,1))), P_old[:, [ii,jj]][inds_not_ij]), axis = 1)
            P_temp_1 = np.round(P_temp_1 / p_ss_temp[0:-1],decimals=4)
            P_temp_2 = np.sum(np.multiply(np.transpose(np.tile(np.r_[p_ss_old[ii],p_ss_old[jj]] , (i-1,1))), P_old[[ii,jj],:][:,inds_not_ij]),axis=0, keepdims=True)
            P_temp_2 = np.round(P_temp_2 / p_ss_temp[-1],decimals=4)
            P_temp_3 = np.sum(np.multiply(np.transpose(np.tile(np.r_[p_ss_old[ii],p_ss_old[jj]], (2,1))), P_old[:,[ii,jj]][[ii,jj], :]))
            P_temp_3 = np.round(P_temp_3 / p_ss_temp[-1],decimals=4)
            
            logP_temp_1 = np.log2(P_temp_1)
            logP_temp_1[logP_temp_1 == -inf] = 0
            logP_temp_2 = np.log2(P_temp_2)
            logP_temp_2[logP_temp_2 == -inf] = 0
            logP_temp_3 = np.log2(P_temp_3)
            logP_temp_3 = np.array([logP_temp_3])
            logP_temp_3[logP_temp_3 == -inf] = 0

            #Compute change in upper bound on mutual information
            d1 = round(-sum(np.multiply(np.multiply(p_ss_temp[0:-1], P_temp_1), logP_temp_1)), 4)
            d2 = round(- p_ss_temp[-1]*np.sum(np.multiply(P_temp_2,logP_temp_2)), 4)
            d3 = round(- p_ss_temp[-1]*P_temp_3*float(logP_temp_3[0]), 4)
            d4 = round(np.sum(np.multiply(p_ss_old, np.multiply(P_old[:,ii], logP_old[:, ii]))), 4)
            d5 = round(np.sum(np.multiply(np.multiply(p_ss_old, P_old[:,jj]), logP_old[:,jj])), 4)
            d6 = round(p_ss_old[ii]*np.sum(np.multiply(P_old[ii,:],logP_old[ii,:])), 4)
            d7 =  round(p_ss_old[jj]*np.sum(np.multiply(P_old[jj,:], logP_old[jj,:])), 4)
            d8 = round(- p_ss_old[ii]*(P_old[ii,ii]*logP_old[ii,ii] + P_old[ii,jj]*logP_old[ii,jj]), 4)
            d9 = round(- p_ss_old[jj]*(P_old[jj,jj]*logP_old[jj,jj] + P_old[jj,ii]*logP_old[jj,ii]), 4)
            dS = d1+d2+d3+d4+d5+d6+d7+d8+d9
            dS = round(dS, 4)
            S_temp = 10000*(S_old + dS)
            
        
               # Keep track of all entropies:
            S_all[:,ind] = S_temp



         # Find minimum entropy:
        [dummy, min_inds] = np.nonzero(S_all == np.min(S_all))
        test1 = list(min_inds)
        min_ind = test1[-1]   #fix
    
        # Save mutual information:
        S_old = S_all[:,min_ind]
        S[:,i-1] = S_old
    
        # Compute old transition probabilities:
        ii_new = int(I[min_ind])
        jj_new = int(J[min_ind])
        
        inds_not_ij = list(range(0,(ii_new))) + list(range((ii_new+1),(jj_new))) + list(range((jj_new+1),(i+1))) 

        p_ss_new = np.append(p_ss_old[inds_not_ij], p_ss_old[ii_new] + p_ss_old[jj_new])
        P_joint = np.multiply(np.transpose(np.tile(p_ss_old, (i+1, 1))), P_old)
        P_joint = np.c_[np.r_[P_joint[:,inds_not_ij][inds_not_ij,:],np.sum(P_joint[[ii_new,jj_new],:][:,inds_not_ij], axis=0, keepdims=True)], 
                         np.append(np.sum((P_joint[:, [ii_new,jj_new]][inds_not_ij]), axis=1),np.sum(P_joint[:,range(ii_new,jj_new+1)][range(ii_new,jj_new+1),:]))]
        
        
        
        P_old = np.divide(P_joint, np.transpose(np.tile(p_ss_new, (i, 1))))
        p_ss_old = p_ss_new
    
        logP_old = np.log2(P_old)
        logP_old[logP_old == -inf] = 0

        # Record clusters and graph:
        clusters[i-1] = list(range(0,ii_new))+list(range(ii_new+1, jj_new))+list(range(jj_new+1, i+2)) 
        clusters[i-1].append([ii_new, jj_new])
        Gs[i-1] = P_joint*2*E
    
        # Compute lower bound on mutual information:
        P_low = np.c_[P_low[:,list(range(0,ii_new))+list(range(ii_new+1, jj_new))+list(range(jj_new+1, i+1))] , P_low[:,ii_new] + P_low[:,jj_new]]

        logP_low = np.log2(P_low)
        logP_low[logP_low == -inf] = 0
        S_low[:,i-1] = np.round(-sum(np.multiply(p_ss , np.sum(np.multiply(P_low, logP_low), axis=1))), decimals=4)

    return(S, S_low, clusters, Gs)


results = rate_distortion(G,1,2)
test2= np.flip(results[1][0])
#Results from Matlab
test3=np.array([2.5810, 2.4936,2.4483,2.4070,2.3553,2.3074,2.2547,2.1888,2.1167,2.0305,1.9632,1.8931,1.8205,1.7568,1.6888,1.6018,1.5205,1.4369,1.3518,1.2655,1.1780,1.0893,1.0026,0.8854,0.7665,0.6734,0.5811,0.4499,0.3956,0.3629,0.3302,0.2917,0.0987,0])

plt.title("Zachary's Karate Club")
plt.xlabel("Distortion")
plt.ylabel("Information rate(bits)")
plt.plot(test2, label = 'Python')
plt.plot(test3, label= 'original')
plt.legend(loc="upper right")
plt.show




