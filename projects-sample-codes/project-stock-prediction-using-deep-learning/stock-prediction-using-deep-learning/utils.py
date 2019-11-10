# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def classify(Y_sigmoid):
    '''
    
    Params:
        Y_sigmoid:　Model predict result, array, shape=(m,1)
        
    Returns:
        Y_logic:    binary predict result, array, shape=(m,1)
    '''

    Y_logic = np.array([[1] if y > 0.5 else [0] for y in Y_sigmoid])
    return Y_logic


def mystat(data, np=True):
    d1 = data.count()
    d2 = data.min()
    d3 = data.quantile(0.5)
    d4 = data.max()
    d5 = data.mean()
    d6 = data.std()
    d7 = data.skew()
    d8 = data.kurt()

    if np:
        stat = np.array([d1, d2, d3, d4, d5, d6, d7, d8])
    else:
        stat = pd.DataFrame([d1, d2, d3, d4, d5, d6, d7, d8],
                            index=['count', 'min', 'median', 'max', 'mean', 'std', 'skew', 'kurt'])

    return stat
