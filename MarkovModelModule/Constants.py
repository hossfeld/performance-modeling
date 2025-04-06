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
                     DEPARTURE: plt.cm.tab10(1)}

COLOR_NODE_DEFAULT = plt.cm.tab20(1) 
COLOR_NODE_BLOCKING = plt.cm.tab20(7) 


def savefig(fname, printLatex=False, returnLatex=True):    
    path = 'C:/git/2025-paperWithMarkovModel/figs/'
    plt.savefig(path+fname+".pdf")
    plt.savefig(path+fname+".svg")
    plt.savefig(path+fname+".png")
    fname = "simple"+fname
    if printLatex or returnLatex:
        img = fname  # Set your image filename
        img_clean = img.replace("_", "")
        latex_string = fr"""\begin{{figure}}[h]
            \centering
            \includegraphics[width=\figwidth]{{{img}}}
            \caption{{Image: {img_clean}}}
            \label{{fig:{img_clean}}}
        \end{{figure}}"""
        if printLatex: 
            print(latex_string)
        else: 
            print(fname)
        return latex_string
    else:
        print(fname) 