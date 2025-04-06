# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 22:26:27 2025

@author: tobia
"""



from MarkovModelModule import StateTransitionGraph # imports class     
import Constants as const

import matplotlib.pyplot as plt  # For plotting
import numpy as np
        
#%% create a transition graph for M/M/1-0
def createMM1(lam=1.0, mu=1.0):
    parameters={'lambda':lam, 'mu':1.0}
    G = StateTransitionGraph(parameters)    
    
    G.addTransition(0, 1, G.sym("lambda"), tid = const.ARRIVAL)    
    G.addTransition(1, 0, G.sym("mu"), tid = const.DEPARTURE)    
    
    # set default labels and colors for nodes
    G.setAllStateDefaultProperties() # labels and colors
    
    return G


#%% create transition graph and compute system characteristics
G = createMM1()
probs = G.calcSteadyStateProbs() # this is required to compute the steady state probabilities


#%% Plotting the graph
pos = {s:(s,0) for s in G.states()}
G.drawTransitionGraph(pos, bended=True, label_pos=0.5, num=2)
#%% Simulate and animate in Graph
ts = np.linspace(0.0,4,30)
X0 = G.getEmptySystemProbs()
#%%
xt = np.zeros((len(G), len(ts)))
for i,t in enumerate(ts):
    probs_t = G.transientProbs(X0, t)
    xt[:,i] = np.array(list(probs_t.values()))
#%%    
plt.figure(1, clear=True)
plt.plot(ts, xt.T, '.-')
plt.xlabel('time (s)')
plt.ylabel('probability')
plt.legend(probs_t.keys(), title='state')
plt.grid(which='major')
#%%

simplified_solution, num_dict  = G.symSolveMarkovModel()
simplified_solution