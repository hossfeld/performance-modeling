# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:01:23 2025

@author: hossfeld
"""
import matplotlib.pyplot as plt  # For plotting

# transistion IDs
ARRIVAL = 1
DEPARTURE = 2

# colors
COLOR_TRANSITIONS = {ARRIVAL: plt.cm.tab10(0),
                     DEPARTURE: plt.cm.tab10(4)}

COLOR_NODE_DEFAULT = plt.cm.tab20(1) 
COLOR_NODE_STALLING = plt.cm.tab20(3) 

 