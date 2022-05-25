# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:04:44 2021

@author: Tobias Hoßfeld
"""
from matplotlib import pyplot as plt
import numpy as np
from discreteTimeAnalysis import *
#%%
A = DU(1,3) # interarrival time
B = DU(5,9)  # service time

#%%
C = max(A,4)

#%%
plt.figure(1, clear=True)
A.plotCDF(label='A', marker='s')
B.plotCDF(label='B')
C.plotCDF(label='max(A,4)')
plt.legend()
#%%
def mymin(*args):
    xmax = max([Ai.xmax for Ai in args])
    xmin = min([Ai.xmin for Ai in args])
    x = np.arange(xmin, xmax+1, dtype=int)
    ccdfs = np.zeros( (len(args),len(x)) )
    for i,Ai in enumerate(args):
        ccdfs[i,:] = 1-Ai.cdf(x)
    
    myccdf = np.prod( ccdfs.clip(0,1), axis=0)
    mycdf = 1-myccdf
    mypmf = np.diff(np.insert(mycdf,0,0))
    return DiscreteDistribution(x,mypmf.clip(0,1))    

def mymax(*args):
    bools = [isinstance(Ai, DiscreteDistribution) for Ai in args]
    if all(bools):
        xmax = max([Ai.xmax for Ai in args])
        xmin = min([Ai.xmin for Ai in args])
        x = np.arange(xmin, xmax+1, dtype=int)
        cdfs = np.zeros( (len(args),len(x)) )
        for i,Ai in enumerate(args):
            cdfs[i,:] = Ai.cdf(x)
        
        mycdf = np.prod( cdfs, axis=0)
        mypmf = np.diff(np.insert(mycdf,0,0))
        return DiscreteDistribution(x,mypmf.clip(0,1))
#%%    
plt.figure(2, clear=True)
A.plotCDF(label=f'A ~ {A.name}', marker='s')
B.plotCDF(label=f'B ~ {B.mean()}')   
 
D = mymax(A,B)
D.plotCDF(label='D')
plt.legend()

#%%
A = DU(2,9)
C = max(A, 4)
D = mymax(A, DET(4))
plt.figure(4, clear=True)
C.plotCDF()
D.plotCDF()