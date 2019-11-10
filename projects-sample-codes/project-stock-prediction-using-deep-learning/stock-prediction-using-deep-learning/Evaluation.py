# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:08:02 2018

@author: Administrator

Evaluation
    Evaluate Model Cost and Accuracy
"""

# %% Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Evaluation(Y_predict, Y_test, Scores=None, Cost=None, Cost_dev=None, Acc=None, Acc_dev=None):
    # predict and Actual Results
    print("\nPredict and Actual Results:")
    plt.plot(range(len(Y_predict)), Y_predict, range(len(Y_test)), Y_test)
    plt.show()

    # e1 Training Process - Test, Dev; Cost, Acc
    if Cost is not None: plt.plot(Cost, label='Cost')
    if Cost_dev is not None: plt.plot(Cost_dev, label='Cost_dev')
    if Acc is not None: plt.plot(Acc, label='Acc')
    if Acc_dev is not None: plt.plot(Acc_dev, label='Acc_dev')

    if Cost or Cost_dev or Acc or Acc_dev:
        plt.legend()
        plt.show()

    # e2 Training Result
    if Scores is not None:
        # Train, Dev, Test; Cost, Acc
        for Label, Score in Scores.items():
            print(Label + " Cost = " + str(Score[0]))
            print(Label + " Accuracy = " + str(Score[1]))

    elif Scores is None:
        # Acc
        precision = np.sum(Y_test == Y_predict) / len(Y_predict)
        print("\nPrecision: \n" + str(precision))
