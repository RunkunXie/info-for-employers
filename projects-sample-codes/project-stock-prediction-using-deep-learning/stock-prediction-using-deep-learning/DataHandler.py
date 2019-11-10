# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:20:02 2018

@author: Administrator

DataHandler
    Read stored csv data, generate training, validation, and test sets.
"""

# %% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
class DataHandler:
    """
    Read stored csv data, generate training, validation, and test sets.

    Usage:
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = DataHandler.GetTrainDevTest(X, Y)
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def Read(self, basic=False, pct=1, nozero=True):
        """
        read csv data, take date at first column as "Index"

        format：
            Date    Col1    Col2    ...
            date1   value1  value2  ...
            ...     ...     ...     ...

        Params:
            basic:  only read basic(Open High Low Close) data, bool
        """

        data = pd.read_csv(self.file_name)

        data.index = pd.to_datetime(data['Date'])
        data.drop("Date", axis=1, inplace=True)

        # adjust input
        '''
        only read basic(Open High Low Close) data
        '''
        if basic:
            data = data.iloc[:, :6]

        # other adjusts
        '''
        Do not need Preclose
        '''
        if "Pre_close" in data.columns:
            data.drop("Pre_close", axis=1, inplace=True)

        # drop nan
        data.dropna(axis=0, how='any', inplace=True)

        # reduce data
        if pct < 1:
            data = data.iloc[:int(data.shape[0] * pct), :]

        # drop equal
        if nozero:
            data = data[data['Close'].diff() != 0]

        return data

    def GetXY(self, data, norm=True, overlap=True, flatten=True, shuffle=True, time_range=5,
              divide_method=[0.98, 0.996, 1.0], nozero=True):
        '''
        generate X and Y based on data
        
        Paras:   
            data:       pd.DataFrame
            shuffle:    shuffle data[optional]
            norm:       normalization [optional]
            time_range: num of days, X[optional]
            overlap:    overlap X[optional]
            flatten:    flatten X[optional]
            nozero:     cases where price do not change [optional]
            
        Returns: (X, Y)
            if flatten:
                X: array, shape=(m_train, T_x, n_x)
                Y: array, shape=(m_train, 1)                            
            if not flatten:
                X: array, shape=(m_train, n_x) -- n_x = T_x * n_x
                Y: array, shape=(m_train, 1)            
        '''

        # Assert
        assert isinstance(data, pd.DataFrame)
        assert norm in [True, False]
        assert overlap in [True, False]
        assert flatten in [True, False]
        assert shuffle in [True, False]
        assert isinstance(time_range, int) and time_range > 1

        # Normalization
        if norm:
            data = (data - data.mean()) / data.std()

        # GetXY
        '''
        iterate data, choose every time_range lines as input, and 
        out-of-range price/change as input. 
        (use '(len(data)-1)' because change need next day's data)
        (use 'len(data)-time_range' when choose overlapping data)
        '''

        XY = []

        if overlap is False:

            sep = int(int((len(data) - 1) / time_range) / 20)
            for i in range(int((len(data) - 1) / time_range)):
                if (data.loc[data.index[(i + 1) * time_range], "Close"] != data.loc[
                    data.index[(i + 1) * time_range - 1], "Close"]) or not nozero:
                    # X
                    if flatten:
                        XY.append(data.iloc[i * time_range:(i + 1) * time_range, :].values.flatten('C').tolist())
                    elif not flatten:
                        XY.append(data.iloc[i * time_range:(i + 1) * time_range, :].values.tolist())

                    # Y
                    XY[-1].append(1 if data.loc[data.index[(i + 1) * time_range], "Close"] \
                                       - data.loc[data.index[(i + 1) * time_range - 1], "Close"] > 0 else 0)

                    # timer
                    if i % sep == 0:
                        print("[" + "=" * int(i / sep) + str(">" if int(i / sep) < 20 else "=") + '-' * (
                                20 - int(i / sep)) + "]")

        elif overlap is True:

            sep = int((len(data) - time_range) / 20)
            for i in range(len(data) - time_range):
                if (data.loc[data.index[i + time_range], "Close"] != data.loc[
                    data.index[i + time_range - 1], "Close"]) or not nozero:
                    # X
                    if flatten:
                        XY.append(data.iloc[i:i + time_range, :].values.flatten('C').tolist())
                    elif not flatten:
                        XY.append(data.iloc[i:i + time_range, :].values.tolist())

                    # Y
                    XY[-1].append(1 if data.loc[data.index[i + time_range], "Close"] \
                                       - data.loc[data.index[i + time_range - 1], "Close"] > 0 else 0)

                    # timer
                    if i % sep == 0:
                        print("[" + "=" * int(i / sep) + str(">" if int(i / sep) < 20 else "=") + '-' * (
                                20 - int(i / sep)) + "]")

        # Shuffle
        '''
        shuffle XY list, make sure X and Y matches
        '''

        if shuffle:
            XY_part = XY[:int(divide_method[0] * len(XY))]
            np.random.shuffle(XY_part)
            XY[:int(divide_method[0] * len(XY))] = XY_part

        # Create X and Y
        '''
        get X and Y from XY, generate X and Y arrays
        '''

        if flatten:
            assert data.shape[1] * time_range == len(XY[0]) - 1
        elif not flatten:
            assert data.shape[1] * time_range == len(XY[0][0]) * (len(XY[0]) - 1)

        X = []
        Y = []

        for i in range(len(XY)):
            X.append(XY[i][:-1])
            Y.append(XY[i][-1])

        X = np.array(X)
        Y = np.array(Y, ndmin=2).T

        return (X, Y)

    def GetTrainDevTest(self, X, Y, divide_method=[0.8, 0.9, 1.0]):
        '''
        generate Train Dev Test data, based on X and Y
        
        Paras: 
            X:              array, shape=(m_train, n_x)
            Y:              array, shape=(m_train, 1)
            divide_method:  list
            
        Returns: 
            (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)            
        '''

        length = len(Y)
        assert len(X) == len(Y)

        divide = [round(length * i) for i in divide_method]

        X_train = X[0:divide[0]]
        Y_train = Y[0:divide[0]]
        X_dev = X[divide[0]:divide[1]]
        Y_dev = Y[divide[0]:divide[1]]
        X_test = X[divide[1]:divide[2]]
        Y_test = Y[divide[1]:divide[2]]

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


# %% Test Class
if __name__ == '__main__':
    file_name = "HS300_5min_basic.csv"

    D = DataHandler(file_name)
    data = D.Read()
    print("data:\n", data.head())
    plt.plot(data.iloc[:, 0].values)

    X, Y = D.GetXY(data)
    print("X:\n", X[0:3])
    print("Y:\n", Y[0:3])

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = D.GetTrainDevTest(X, Y)
