# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:58:56 2018

Define two functions for use

@author: 
"""

import math
from scipy.stats import norm
e = math.e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#%%
def EuropeanOptions(S, K, T, sigma, r, otype):
    
    d1 = 1 / (sigma*T**(0.5)) * (np.log(S/K)+(r+sigma**2/2)*T)
    d2 = d1 - sigma*T**(0.5)
    
    c = norm.cdf(d1)*S - norm.cdf(d2)*K*math.e**(-r*T)
    p = - norm.cdf(-d1)*S + norm.cdf(-d2)*K*math.e**(-r*T)
    
    # otype == 1 if the option is a call; otype == 0 if the option is a put
    return c * otype + p * (1 - otype)
    


def EW_all(l, beta):
    
    n = l.shape[0]
    nvars = l.shape[1]

    ewma = np.array(l).reshape(l.shape + (1,))
    ewma_cross = np.ones((n, nvars, nvars))
    
    for i in range(n):
        ewma_cross[i] = np.dot(ewma[i],ewma[i].T)
        if i == 0:
            ewma[i] = (1 - beta) * ewma[i]
            ewma_cross[i] = (1 - beta) * ewma_cross[i]
        else:
            ewma[i] = beta * ewma[i-1] + (1 - beta) * ewma[i]
            ewma_cross[i] = beta * ewma_cross[i-1] + (1 - beta) * ewma_cross[i]
    
    correct = np.array([1-beta**(t+1) for t in range(n)]).reshape(n,1)
    
    ewma_correct = ewma 
    ewma_cross_correct = ewma_cross 

    for i in range(n):
        ewma_correct[i] = ewma_correct[i] / correct[i]
        ewma_cross_correct[i] = ewma_cross_correct[i] / correct[i]
    
    return ewma_correct, ewma_cross_correct


