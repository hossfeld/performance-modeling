# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:42:09 2019

@author: hossi
"""

#%%
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats 
import pandas as pd 
from tqdm import tqdm

#%%
x = np.arange(10)
plt.figure(1)
plt.clf()
for p,m in zip((0.05, 0.1, 0.25, 0.5),('s','d','o','x','.')):
    q=1-p
    plt.plot(x, (1-p)**x*p, '-', marker=m, label=f'p={p:.2f}, E[X]={q/(1-q):.1f}')
plt.xlabel('i')
plt.ylabel('probability $P(X=i)$')
plt.legend()
plt.yscale('log')
plt.tight_layout()
my_dpi, my_fontsize = 600, 20
plt.rcParams.update({'font.size': my_fontsize})
plt.savefig('geomPDF3.svg', dpi=my_dpi)  

#%%
z = np.linspace(0,1,101)
plt.figure(2)
plt.clf()
for p,m in zip((0.25, 0.5, 0.75),('s','d','o')):
    q=1-p
    plt.plot(z, (1-q)/(1-q*z), '-', label=p)
plt.xlabel('i')
plt.ylabel('$X_{EF}(z)$')
plt.legend(title='succ. prob. p')

plt.tight_layout()
my_dpi, my_fontsize = 600, 20
plt.rcParams.update({'font.size': my_fontsize})
plt.savefig('geomErzeugendeFunktion.svg', dpi=my_dpi)  
#%%
x = np.arange(1,30)
x = np.linspace(0,1,101)
plt.figure(3)
plt.clf()
plt.plot(x, 1/np.sqrt(x), 'd-')
plt.xlabel('k')
plt.ylabel('$c_A$')

plt.tight_layout()
my_dpi, my_fontsize = 600, 20
plt.rcParams.update({'font.size': my_fontsize})
#plt.savefig('cA-erlangk.svg', dpi=my_dpi)  

#%% 
x = np.random.randint(1,6,size=50)
plt.figure(4)
plt.clf()
plt.plot(np.arange(len(x)), x, 's')
plt.xlabel('measurement i')
plt.ylabel('value $x_i$')
plt.tight_layout()
my_dpi, my_fontsize = 600, 20
plt.rcParams.update({'font.size': my_fontsize})
plt.savefig('randDU.svg', dpi=my_dpi)  
#%%
y = np.tile(np.arange(1,6), 10)
z = np.arange(1,6).repeat(10)
plt.figure(5)
plt.clf()
plt.plot(np.arange(len(y)), y, 's', label='M1')
plt.plot(np.arange(len(z)), z, 'd', label='M2')
#plt.plot(np.arange(len(x)), x, 'd', label='measurement 3')

plt.xlabel('measurement i')
plt.ylabel('value $y_i$')
plt.legend()
plt.tight_layout()
my_dpi, my_fontsize = 600, 20
plt.rcParams.update({'font.size': my_fontsize})
plt.savefig('pattern.svg', dpi=my_dpi)  