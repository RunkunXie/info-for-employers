# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:38:07 2018

@author: Administrator

Stock Prediction 
    Main Program
    1 Access data
    2 Make prediction
    3 Evaluate results
    4 Backtest Strategy
    5 Strategy Summary
"""

# %% Import Packages
# Classes
from DataHandler import DataHandler
from Model import My_Model
from utils import *

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Packages
from sklearn import linear_model  # linear model
import tensorflow as tf

# Keras
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, LSTM, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

# %% Choose Model
model_choices = {0: 'Logistic', 1: 'SimpleNN', 2: 'DNN', 3: 'RNN', 4: 'LSTM'}
print(model_choices)
model_select = 1  # input("Choose you model:")
model_type = model_choices[int(model_select)]

# %% 1. Access Data
# Read Data
print("Read Data")
part = 2
IX = 'IC'
frequency = 'min'
indicator = 'derive'

file_name = "./data/" + IX + "_" + frequency + "_" + indicator + (
    "_" + str(part) if frequency == 'min' else '') + ".csv"

basic = False
D = DataHandler(file_name)
data = D.Read(basic, pct=1)

# create X and Y
print("create X and Y")
norm = True
overlap = True
flatten = False if model_type in ['RNN', 'LSTM'] else True
shuffle = False if part == 2 else True
time_range = 10

X, Y = D.GetXY(data, norm, overlap, flatten, shuffle, time_range, divide_method=[1, 0.996, 1.0], nozero=True)

# create Train/Dev/Test Sets
if len(Y) < 10000:
    divide_method = [0.7, 0.85, 1]
elif len(Y) < 50000:
    divide_method = [0.90, 0.95, 1.0]  #
elif len(Y) < 150000:
    divide_method = [0.96, 0.98, 1.0]  #

(X_train, Y_train, X_dev, Y_dev, X_test, Y_test) = D.GetTrainDevTest(X, Y, divide_method)

# %% 2. Model

model = My_Model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

#############################################################################
if model_type == 'Logistic':
    ## IC min
    # 541 561 521 50 ep - ns p2 de1 conv
    batch_size = 2048
    epochs = 50
    model.My_Logistic()
    model.My_Optimizer(0.005)

#############################################################################
if model_type == 'SimpleNN':
    ## IC min
    # 544 562 524 50 epoch - ns p2 de1 - conv
    hidden_layer_dims = [64]
    dropout_probs = [0]
    batch_size = 2048
    epochs = 100
    model.My_NN(hidden_layer_dims, dropout_probs)
    model.My_Optimizer(0.05)

#############################################################################
if model_type == 'DNN':
    # IC min
    # 546 563 524 20 epochs depre1
    hidden_layer_dims = [64, 64, 16]
    dropout_probs = [0.2, 0.2, 0.1]
    batch_size = 2048
    epochs = 100
    model.My_NN(hidden_layer_dims, dropout_probs)
    model.My_Optimizer(0.005)

#############################################################################
if model_type == 'RNN':
    # IC min
    # 539 575 511 25 epochs de1
    hidden_layer_dims = [16, 16, 8]
    dropout_probs = [0.2, 0.2, 0.1]
    batch_size = 2048
    epochs = 50
    model.My_RNN(hidden_layer_dims, dropout_probs)
    model.My_Optimizer(0.008)

#############################################################################
if model_type == 'LSTM':
    ## IC

    # 546 59 52 50 epoch -  ns p2 depre1 - not conv overfit
    hidden_layer_dims = [16, 16, 8]
    dropout_probs = [0.1, 0.1, 0.0]
    batch_size = 2048
    epochs = 20
    model.My_LSTM(hidden_layer_dims, dropout_probs)
    model.My_Optimizer(0.05)

# %% 3. Evaluate
model.My_Compile_for_evaluate()

model.My_Fit(batch_size=batch_size, epochs=epochs)

model.My_Predict()
model.My_Evaluate()

version = 'de1'

my_path_model = './output/my' + frequency + '_' + model_type + ('_s' if shuffle else '_ns') + '_p' + str(
    part) + '_' + IX + '_' + version + '.h5'
my_path_hist = my_path_model[:-3] + '.csv'
my_path_data = my_path_model[:-3] + '.xlsx'

# %% Save and Read Model
if True:
    # read
    model.model = load_model(my_path_model)

    # read
    hist = pd.read_csv(my_path_hist, index_col=0)
    model.Cost = hist['loss'].tolist()
    model.Cost_dev = hist['val_loss'].tolist()
    model.Acc = hist['acc'].tolist()
    model.Acc_dev = hist['val_acc'].tolist()

    model.My_Predict()
    model.My_Evaluate()

    # write
    write = pd.ExcelWriter(my_path_data)
    model.Scores.to_excel(write, sheet_name='Scores')
    model.ConfusionMatrces.to_excel(write, sheet_name='ConfusionMatrces')
    model.F1_measures.to_excel(write, sheet_name='F1_measures')
    hist.to_excel(write, sheet_name='hist')
    model.ROC.T.to_excel(write, sheet_name='ROC')
    write.save()

if True:
    # save
    model.model.save(my_path_model)

    # save
    hist_columns = ['loss', 'val_loss', 'acc', 'val_acc']
    hist = pd.DataFrame(np.array([model.Cost, model.Cost_dev, model.Acc, model.Acc_dev]).T, columns=hist_columns)
    hist.to_csv(my_path_hist)

    # write
    write = pd.ExcelWriter(my_path_data)
    model.Scores.to_excel(write, sheet_name='Scores')
    model.ConfusionMatrces.to_excel(write, sheet_name='ConfusionMatrces')
    model.F1_measures.to_excel(write, sheet_name='F1_measures')
    hist.to_excel(write, sheet_name='hist')
    model.ROC.T.to_excel(write, sheet_name='ROC')
    write.save()


# %% 4. Evaluate Strategy
def strategy(price_dev, Y_predict_dev, ls, label, yu=0.5, bstype='b'):
    change_dev = np.diff(price_dev)

    # RANDOM
    #    return_dev = change_dev / price_dev[:-1]
    #
    #    RAN = np.random.rand(len(return_dev))
    #    return_dev[RAN<0.5] = 0
    #    cum_return_dev = np.cumprod(return_dev+1)
    #    plt.plot(cum_return_dev)

    # LSTM
    return_dev = change_dev / price_dev[:-1]

    if bstype == 'b':
        return_dev[Y_predict_dev <= yu] = 0
    elif bstype == 's':
        return_dev[Y_predict_dev > yu] = 0
        return_dev[Y_predict_dev <= yu] *= -1
    elif bstype == 'bs':
        return_dev[Y_predict_dev <= yu] *= -1

    cum_return_dev = np.cumprod(return_dev + 1)
    plt.plot(cum_return_dev, ls, label=label)

    return cum_return_dev


predict_dev_data_pd = pd.read_excel('./output_final/output_final_predict.xlsx')

################

fig = plt.figure(figsize=(12, 10))

for i, c in enumerate(predict_dev_data_pd.columns):
    plt.subplot(5, 1, i + 1)
    plt.plot(predict_dev_data_pd[c])
    plt.plot(range(len(predict_dev_data_pd[c])), 0.5 * np.ones(len(predict_dev_data_pd[c])))
    plt.ylim([0.35, 0.65])
    plt.title(c)

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.5, wspace=0.3)
plt.show()

################


################
ls = ['-', '--', '-.', ':', '-', '']

data_dev = data.iloc[-len(Y_test) - len(Y_dev) - 1:-len(Y_test), :]
price_dev = data_dev['Close'].values

change_dev = np.diff(price_dev)
return_dev = change_dev / price_dev[:-1]
cum_return_dev_bh = np.cumprod(return_dev + 1)
plt.plot(cum_return_dev_bh, ls[5], label='B&H')

cum_return_dev_all = cum_return_dev_bh

for i, c in enumerate(predict_dev_data_pd.columns):
    cum_return_dev = strategy(price_dev, predict_dev_data_pd[c], ls[i], label=c, bstype='bs')
    cum_return_dev_all = np.vstack((cum_return_dev_all, cum_return_dev))

plt.legend()
plt.show()
################


#################
#
# data_dev = data.iloc[-len(Y_test)-len(Y_dev)-1:-len(Y_test),:]
# price_dev = data_dev['Close'].values
#
# dvd = [1, int(len(price_dev)/2), len(price_dev)]
#
#
# fig=plt.figure(figsize=(12,8))
#
# for i in range(2):
#    plt.subplot(2,2,i+1)
#    
#    data_dev = data.iloc[-len(Y_test)-len(Y_dev)-1:-len(Y_test),:]
#    price_dev = data_dev['Close'].values
#    
#    price_dev = price_dev[dvd[i]-1:dvd[i+1]]
#    
#    change_dev = np.diff(price_dev) 
#    return_dev = change_dev / price_dev[:-1]
#    cum_return_dev = np.cumprod(return_dev+1)
#    plt.plot(cum_return_dev,label='B&H')
#    
#    for c in predict_dev_data_pd.columns:
#        strategy(price_dev, predict_dev_data_pd[c][dvd[i]-1:dvd[i+1]-1], label=c,bstype='bs')
#    
#    plt.legend(loc=2)
#        
# plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2,wspace=0.3)
# plt.show()
#
#################

# %% 5. Strategy Summary

indexes = ['B&H', 'Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM']
columns = ['cum_return', 'ann_return', 'vol', 'sharpe']

evaluate_all = np.zeros((len(indexes), len(columns)))
evaluate_all[:] = np.nan


def cum_return(cum_return_dev_all):
    return cum_return_dev_all[:, -1] - 1


def ann_return(cum_return_dev_all):
    cum = cum_return(cum_return_dev_all)
    ann = (cum + 1) ** (252 * 4 * 60 / cum_return_dev_all.shape[1]) - 1

    return ann


def sigma(cum_return_dev_all):
    return np.std(cum_return_dev_all, axis=1) * np.sqrt(252 * 4 * 60)


def sharpe(cum_return_dev_all):
    ann = ann_return(cum_return_dev_all)
    sigmaP = sigma(cum_return_dev_all)

    return ann / sigmaP


evaluate_all[:, 0] = cum_return(cum_return_dev_all)
evaluate_all[:, 1] = ann_return(cum_return_dev_all)
evaluate_all[:, 2] = sigma(cum_return_dev_all)
evaluate_all[:, 3] = sharpe(cum_return_dev_all)

evaluate_all_pd = pd.DataFrame(evaluate_all, index=indexes, columns=columns)

print(evaluate_all_pd)

write = pd.ExcelWriter('./output_final/output_final_evaluate.xlsx')
evaluate_all_pd.to_excel(write)
write.save()
