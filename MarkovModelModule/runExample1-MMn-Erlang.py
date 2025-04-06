# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:50:33 2025

@author: tobia
"""


from MarkovModelModule import StateTransitionGraph # imports class     
import Constants as const

import matplotlib.pyplot as plt  # For plotting
import numpy as np
        
#%% create a transition graph for M/M/n-0
def createMMn(lam=1.0, mu=1.0, n=5):
    parameters={'lambda':lam, 'mu':1.0, 'n':5}
    G = StateTransitionGraph(parameters)
    
    # define transitions    
    for i in range(n):
        G.addTransition(i, i+1, G.sym("lambda"), tid = const.ARRIVAL)    
        G.addTransition(i+1, i, (i+1)*G.sym("mu"), tid = const.DEPARTURE)    
    
    # set default labels and colors for nodes
    G.setAllStateDefaultProperties() # labels and colors
    G.setStateColor(G.n, const.COLOR_NODE_BLOCKING)  # blocking

    return G

# compute system characteristics based on steady state probs
def getSystemCharacteristicsSimple(G):    
    res = {"blocking_prob": G.prob(G.n), "idle_prob": G.prob(0), "expectedNumbers": np.sum( [s*G.prob(s) for s in G.states()] )}    
    return res

#%% create transition graph and compute system characteristics
G = createMMn()
probs = G.calcSteadyStateProbs() # this is required to compute the steady state probabilities
res = getSystemCharacteristicsSimple(G)
for k in res:
    print(f"{k} = {res[k]:.4f}")


#probs = np.array(list(G.state_probabilities.values())) # state_probabilities is a dict
#probs = G.state_probabilities

#%%
plt.figure(1, clear=True)
plt.bar(probs.keys(), probs.values())
plt.xlabel('states')
plt.ylabel('probability')
plt.grid(axis='y', which='major')


#%% Plotting the graph
pos = {s:(s,0) for s in G.states()}
G.drawTransitionGraph(pos, bended=True, label_pos=0.5, num=2)
#%% Simulate and animate in Graph
states, times = G.simulateMarkovModel(startNode=0, numEvents=2000)
G.animateSimulation(expectedTimePerState=1, states=states, times=times, pos=pos, bended=True, num=2 )
