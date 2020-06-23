# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:25:12 2018

@author: Administrator

My_Model
    implemented following models:
        1 Logistic Regression
        2 Basic Neural Network, ANN
        3 Recurrent Neural Network, RNN
        4 Long Short Term Memory Network, LSTM
"""
# %% Import Packages
# Classes
from DataHandler import DataHandler
from Evaluation import Evaluation
from utils import *

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Packages
from sklearn import linear_model  # linear model
from sklearn.metrics import roc_curve, auc  # roc
import tensorflow as tf

# keras
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM, GRU, SimpleRNN, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

np.random.seed(4)


# %% Model for prediction
class My_Model:
    """
    Model Base Class
    """

    def __init__(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.X_test = X_test
        self.Y_test = Y_test

        self.hist = None

        self.Cost = []
        self.Cost_dev = []
        self.Acc = []
        self.Acc_dev = []

    def My_Logistic(self):

        self.model = Sequential()

        n_x = self.X_train.shape[1]

        self.model.add(Dense(1, activation='sigmoid', input_dim=n_x))

    def My_NN(self, hidden_layer_dims, dropout_probs):

        assert len(hidden_layer_dims) == len(dropout_probs)

        self.model = Sequential()

        n_x = self.X_train.shape[1]
        num_layers = len(hidden_layer_dims)

        for i in range(num_layers):
            if i == 0:
                self.model.add(Dense(hidden_layer_dims[i], activation='relu', input_dim=n_x))
            else:
                self.model.add(Dense(hidden_layer_dims[i], activation='relu'))

            self.model.add(BatchNormalization(axis=-1))
            self.model.add(Dropout(dropout_probs[i]))

        self.model.add(Dense(1, activation='sigmoid'))

    def My_RNN(self, hidden_layer_dims, dropout_probs):

        assert len(hidden_layer_dims) == len(dropout_probs)

        self.model = Sequential()

        T_x = self.X_train.shape[1]
        n_x = self.X_train.shape[2]
        num_layers = len(hidden_layer_dims)

        for i in range(num_layers):
            not_last_layer = True if i != num_layers - 1 else False  # returns a sequence of vectors or a single vector of dimension n_x

            if i == 0:
                self.model.add(SimpleRNN(hidden_layer_dims[i], return_sequences=not_last_layer, input_shape=(T_x, n_x)))
            else:
                self.model.add(SimpleRNN(hidden_layer_dims[i], return_sequences=not_last_layer))

            self.model.add(BatchNormalization(axis=-1))
            self.model.add(Dropout(dropout_probs[i]))

        self.model.add(Dense(1, activation='sigmoid'))

    def My_LSTM(self, hidden_layer_dims, dropout_probs):

        assert len(hidden_layer_dims) == len(dropout_probs)

        self.model = Sequential()

        T_x = self.X_train.shape[1]
        n_x = self.X_train.shape[2]
        num_layers = len(hidden_layer_dims)

        for i in range(num_layers):
            not_last_layer = True if i != num_layers - 1 else False  # returns a sequence of vectors or a single vector of dimension n_x

            if i == 0:
                self.model.add(LSTM(hidden_layer_dims[i], return_sequences=not_last_layer, input_shape=(T_x, n_x)))
            else:
                self.model.add(LSTM(hidden_layer_dims[i], return_sequences=not_last_layer))

            self.model.add(BatchNormalization(axis=-1))
            self.model.add(Dropout(dropout_probs[i]))

        self.model.add(Dense(1, activation='sigmoid'))

    def My_Optimizer(self, lr=0.003, beta_1=0.9, beta_2=0.999, decay=0.01):
        self.optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def My_Metrics(self):
        '''
        use np.array Metrics
        
        t: target
        o: output
        '''

        t = self.Y_dev.copy()
        o = self.Y_predict_dev.copy()
        eps = 1e-06
        plt.plot(o)
        # binary_crossentropy: - (t*np.log(o+eps) + (1-t)*np.log(1-o+eps)) / len(t)

        binary_crossentropy = - np.sum(t * np.log(o + eps) + (1 - t) * np.log(1 - o + eps)) / len(t)
        acc = (np.sum(t[o > 0.5] == 1) + np.sum(t[o < 0.5] == 0)) / len(t)

        # print(binary_crossentropy, acc)

    def My_Compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

    def My_Compile_for_evaluate(self):
        '''
        binary_crossentropy / logloss
        
        MSE: Mean Squared Error
        RMSE: Root Mean Squared Error
        MSLE: Mean Squared Logarithmic Error
        
        MAE: Mean Absolute Error
        MAPE: Mean Absolute Percentage Error
         
        Confusion Matrix: TP, FP, TN, FN
        P: Precision = TP/(TP+FP)
        R: Recall = TP/(TP+FN) = TPR
        F1: F1-measure = 2/((1/P)+(1/R)) = (2*P*R)/(P+R)

        TPR: True Positive Rate = TP/(TP+FN)=TP/actual positives
        FPR: False Positive Rate = FP/(FP+TN)=FP/actual negatives
        ROC: Receiver Operating Characteristic
        AUC: Area Under Curve
        
        R^2
        
        '''

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy', 'mse', 'msle', 'mae', 'mape'])

    def My_Fit(self, batch_size=32, epochs=5):

        self.hist = self.model.fit(self.X_train, self.Y_train,
                                   batch_size=batch_size, epochs=epochs,
                                   validation_data=(self.X_dev, self.Y_dev))

    def My_Predict(self):
        self.Y_predict_train_classes = self.model.predict_classes(self.X_train)
        self.Y_predict_dev_classes = self.model.predict_classes(self.X_dev)
        self.Y_predict_test_classes = self.model.predict_classes(self.X_test)

        self.Y_predict_train = self.model.predict(self.X_train)
        self.Y_predict_dev = self.model.predict(self.X_dev)
        self.Y_predict_test = self.model.predict(self.X_test)

    def My_Evaluate(self):
        '''
        Evaluate Model Result:
            1. Cost
            2. Accuracy
            3. Make Graph
        '''

        # Current Model
        print("\nCurrent Model:")
        self.model.summary()

        # e1 - Changing Cost & Acc, of Train & Dev
        print("\nTraining Process:")
        if self.hist is not None:
            self.Cost.extend(self.hist.history['loss'])
            self.Cost_dev.extend(self.hist.history['val_loss'])
            self.Acc.extend(self.hist.history['acc'])
            self.Acc_dev.extend(self.hist.history['val_acc'])

        if self.Cost is not None: plt.plot(self.Cost, label='Cost')
        if self.Cost_dev is not None: plt.plot(self.Cost_dev, label='Cost_dev')
        plt.legend()
        plt.show()

        if self.Acc is not None: plt.plot(self.Acc, label='Acc')
        if self.Acc_dev is not None: plt.plot(self.Acc_dev, label='Acc_dev')
        plt.legend()
        plt.show()

        # e2 - evaluate funcs: 'Cost','Accuracy','mse','msle','mae','mape'

        Score_train = self.model.evaluate(self.X_train, self.Y_train, batch_size=int(self.X_train.shape[0] / 320) * 32)
        Score_dev = self.model.evaluate(self.X_dev, self.Y_dev)
        Score_test = self.model.evaluate(self.X_test, self.Y_test)

        self.Scores = pd.DataFrame([Score_train, Score_dev, Score_test], \
                                   index=['Train Set', 'Cross Validation Set', 'Test Set'], \
                                   columns=['Cost', 'Accuracy', 'mse', 'msle', 'mae', 'mape'])

        # e3 - Confusion Matrix, P, R, F1

        def ConfusionMatrix(t, oc):

            TP = np.sum(t[oc == 1] == 1)
            FP = np.sum(t[oc == 1] == 0)
            TN = np.sum(t[oc == 0] == 0)
            FN = np.sum(t[oc == 0] == 1)

            P_P = TP / (TP + FP)
            R_P = TP / (TP + FN)
            F1_P = (2 * P_P * R_P) / (P_P + R_P)

            P_N = TN / (TN + FN)
            R_N = TN / (TN + FP)
            F1_N = (2 * P_N * R_N) / (P_N + R_N)

            return [TP, FP, TN, FN], [P_P, R_P, F1_P, P_N, R_N, F1_N]

        ConfusionMatrix_train, F1_train = ConfusionMatrix(self.Y_train, self.Y_predict_train_classes)
        ConfusionMatrix_dev, F1_dev = ConfusionMatrix(self.Y_dev, self.Y_predict_dev_classes)
        ConfusionMatrix_test, F1_test = ConfusionMatrix(self.Y_test, self.Y_predict_test_classes)

        self.ConfusionMatrces = pd.DataFrame([ConfusionMatrix_train, ConfusionMatrix_dev, ConfusionMatrix_test], \
                                             index=['Train Set', 'Cross Validation Set', 'Test Set'], \
                                             columns=['TP', 'FP', 'TN', 'FN'])

        self.F1_measures = pd.DataFrame([F1_train, F1_dev, F1_test], \
                                        index=['Train Set', 'Cross Validation Set', 'Test Set'], \
                                        columns=['P_P', 'R_P', 'F1_P', 'P_N', 'R_N', 'F1_N'])

        # e4 - ROC, AUC

        print('\nROC_AUC:')

        def ROC_AUC(t, os):
            fpr, tpr, thresholds = roc_curve(t + 1, os, pos_label=2)
            roc_auc = auc(fpr, tpr)

            return [fpr, tpr, thresholds, roc_auc]

        self.ROC_AUC_train = ROC_AUC(self.Y_train, self.Y_predict_train)
        self.ROC_AUC_dev = ROC_AUC(self.Y_dev, self.Y_predict_dev)
        self.ROC_AUC_test = ROC_AUC(self.Y_test, self.Y_predict_test)

        indexes = pd.MultiIndex.from_tuples(
            [(x, y) for x in ['Train Set', 'Cross Validation Set', 'Test Set'] for y in ['FPR', 'TPR', 'thresholds']])
        self.ROC = pd.DataFrame(self.ROC_AUC_train[:-1] + self.ROC_AUC_dev[:-1] + self.ROC_AUC_test[:-1], index=indexes)
        self.Scores['AUC'] = [self.ROC_AUC_train[-1], self.ROC_AUC_dev[-1], self.ROC_AUC_test[-1]]

        for roc in [self.ROC_AUC_train, self.ROC_AUC_dev, self.ROC_AUC_test]:
            fpr, tpr, thresholds, roc_auc = roc

            plt.figure()
            lw = 2
            plt.figure(figsize=(3, 3))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        print('\nConfusionMatrces:')
        print(self.ConfusionMatrces)

        print('\nF1-measures:')
        print(self.F1_measures)

        print('\nFinan Results:')
        print(self.Scores)

    #        # Predict Result
    #        print("\nPredict and Actual Results:")
    #        plt.plot(range(len(self.Y_predict_train_classes)),self.Y_predict_train_classes,range(len(self.Y_train)),self.Y_train)
    #        plt.show()
    #        plt.plot(range(len(self.Y_predict_test_classes)),self.Y_predict_test_classes,range(len(self.Y_test)),self.Y_test)
    #        plt.show()
