# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:50:33 2025

@author: tobia
"""


from MarkovModelModule import StateTransitionGraph # imports class     
import ConstantsVoD as const

import matplotlib.pyplot as plt  # For plotting
import numpy as np
        
#%% create a transition graph for M/M/n-0
def createVoDMarkovModel(lam=1.0, mu=1.5, n=10, tau=5):
    parameters={'lambda':lam, 'mu':mu, 'n':n, 'tau':tau} # n states
    G = StateTransitionGraph(parameters, "ConstantsVoD")
    
    for i in range(G.tau+1):
        G.addState( (i,0), color=const.COLOR_NODE_STALLING)
    for i in range(1,G.n+1):
        G.addState( (i,1), color=const.COLOR_NODE_DEFAULT)
    
    # define transitions    
    for i in range(G.n):
        if i<tau:
            G.addTransition((i,0), (i+1,0), G.sym_lambda, tid = const.ARRIVAL)    
        elif i==tau:
            G.addTransition((i,0), (i+1,1), G.sym_lambda, tid = const.ARRIVAL)    
        if i>0:
            G.addTransition((i,1), (i+1,1), G.sym_lambda, tid = const.ARRIVAL)    
                
        if i>0:
            G.addTransition((i+1,1), (i,1), G.sym_mu, tid = const.DEPARTURE)                    
        else:
            G.addTransition((i+1,1), (i,0), G.sym_mu, tid = const.DEPARTURE)                            

    return G

# compute system characteristics based on steady state probs
def getSystemCharacteristicsSimple(G):    
    stalling_ratio = np.sum([ G.prob( (x,0)) for x in range(G.tau)])
    stall_frequency = G.mu*G.prob((1,1))
    res = {"stall_ratio": stalling_ratio, "stall_frequency":stall_frequency }    
    return res

#%% create transition graph and compute system characteristics
G = createVoDMarkovModel()

probs = G.calcSteadyStateProbs() # this is required to compute the steady state probabilities

#%%
res = getSystemCharacteristicsSimple(G)
for k in res:
    print(f"{k} = {res[k]:.4f}")

#%%
plt.figure(1, clear=True, figsize=(12,6))

x = np.arange(len(G))
y = [G.prob(s) for s in G]
cols = [G.nodes[s]["color"] for s in G]

plt.bar(x,y, color=cols, label='numerical')
plt.xticks(x, labels=G.states())
plt.xlabel('state')
plt.ylabel('probability')
plt.grid(axis='y', which='major')

simplified_solution, num_dict  = G.symSolveMarkovModel()
plt.plot(x, [num_dict[s] for s in G], 'o', label='symbolic' )
plt.legend()

#%% Plotting the graph
pos = {(x,y):(x,y) for (x,y) in G.states()}
G.drawTransitionGraph(pos, bended=True, label_pos=0.5, num=2)
plt.tight_layout()
#%% Simulate and animate in Graph
states, times = G.simulateMarkovModel(startNode=(0,0), numEvents=2000)
#G.animateSimulation(expectedTimePerState=0.2, states=states, times=times, pos=pos, bended=True, num=2 )
#%%
