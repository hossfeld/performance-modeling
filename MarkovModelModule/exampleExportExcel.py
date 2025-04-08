# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:05:45 2025

@author: tobia
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:50:33 2025

@author: tobia
"""


from MarkovModelModule import StateTransitionGraph # imports class     
import Constants as const

        
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

#%% create transition graph and compute system characteristics
G = createMMn()
pos = {s:(s,0) for s in G.states()}
G.drawTransitionGraph(pos, bended=True, label_pos=0.5, num=1)
#%%
G.exportTransitionsToExcel(open_excel=True)