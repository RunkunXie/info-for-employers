# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:24:04 2018

@author: Administrator

Access Model Result
"""

# %%

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# %% support funcs

frequency = {'Day': 'day', '3 Min': '3min', '5 Min': '5min', '1 Min': ''}
IX = ['HS300', 'ZZ500', 'SH50', 'IF', 'IC']
model_type = {0: 'Logistic', 1: 'SimpleNN', 2: 'DNN', 3: 'SimpleRNN', 4: 'LSTM'}

f = frequency['Day']
ix = IX[1]
mt = model_type[4]

my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

Scores = pd.read_excel(my_path_data, sheetname=0)
ConfusionMatrces = pd.read_excel(my_path_data, sheetname=1)
F1_measures = pd.read_excel(my_path_data, sheetname=2)
hist = pd.read_excel(my_path_data, sheetname=3)
ROC = pd.read_excel(my_path_data, sheetname=4)

# %% 2-1-1 Cost Acc

################################################## day train

f = frequency['Day']
fig = plt.figure(figsize=(12, 9))
lw = 2

for i, ix in enumerate(IX):
    for j, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'
        hist = pd.read_excel(my_path_data, sheetname=3)

        ls = ['+-', '-', '--', '-.', ':'][i]

        plt.subplot(2, 3, j + 1)
        plt.plot(hist['loss'], ls, lw=lw, label=ix)
        plt.legend(loc="lower right")
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.55, 0.85])
        plt.ylabel('Cost', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

        plt.subplot(2, 3, j + 4)
        plt.plot(hist['acc'], ls, lw=lw, label=ix)
        plt.legend(loc="upper right")  # loc=1 
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.45, 0.75])
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2, wspace=0.3)
plt.show()

# fig.savefig('./images/2-1-1-daytrain.png')

##################################################


################################################## 5 min train

f = frequency['5 Min']

fig = plt.figure(figsize=(12, 9))
lw = 2

for i, ix in enumerate(IX):
    for j, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'
        hist = pd.read_excel(my_path_data, sheetname=3)

        ls = ['+-', '-', '--', '-.', ':'][i]

        plt.subplot(2, 3, j + 1)
        plt.plot(hist['loss'], ls, lw=lw, label=ix)
        plt.legend(loc="lower right")
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.55, 0.85])
        plt.ylabel('Cost', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

        plt.subplot(2, 3, j + 4)
        plt.plot(hist['acc'], ls, lw=lw, label=ix)
        plt.legend(loc="upper right")  # loc=1 
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.45, 0.75])
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2, wspace=0.3)
plt.show()
# fig.savefig('./images/2-1-1-5mintrain.png')

##################################################


################################################## 3 min train

f = frequency['3 Min']

fig = plt.figure(figsize=(12, 9))
lw = 2

for i, ix in enumerate(IX):
    for j, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'
        hist = pd.read_excel(my_path_data, sheetname=3)

        ls = ['+-', '-', '--', '-.', ':'][i]

        plt.subplot(2, 3, j + 1)
        plt.plot(hist['loss'], ls, lw=lw, label=ix)
        plt.legend(loc="lower right")
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.55, 0.85])
        plt.ylabel('Cost', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

        plt.subplot(2, 3, j + 4)
        plt.plot(hist['acc'], ls, lw=lw, label=ix)
        plt.legend(loc="upper right")  # loc=1 
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.45, 0.75])
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2, wspace=0.3)
plt.show()
# fig.savefig('./images/2-1-1-3mintrain.png')

##################################################


################################################## 1 min train

f = frequency['1 Min']

fig = plt.figure(figsize=(12, 9))
lw = 2

for i, ix in enumerate(IX):
    for j, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'
        hist = pd.read_excel(my_path_data, sheetname=3)

        ls = ['+-', '-', '--', '-.', ':'][i]

        plt.subplot(2, 3, j + 1)
        plt.plot(hist['loss'], ls, lw=lw, label=ix)
        plt.legend(loc="lower right")
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.55, 0.85])
        plt.ylabel('Cost', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

        plt.subplot(2, 3, j + 4)
        plt.plot(hist['acc'], ls, lw=lw, label=ix)
        plt.legend(loc="upper right")  # loc=1 
        plt.xlim([0.0, 100.0]) if j == 0 else plt.xlim([0.0, 100.0])
        plt.ylim([0.45, 0.75])
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title(['Logistic', 'SimpleNN', 'LSTM'][j] + ' Model')

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2, wspace=0.3)
plt.show()
fig.savefig('./images/2-1-1-mintrain.png')

##################################################

# %% 2-2 F-measure

my_path_output = './output_final/output_final_2-3.xlsx'

data = np.zeros((10, 6 * 3))
data[:] = np.nan

f = frequency['Day']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 6

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        F1_measures = pd.read_excel(my_path_data, sheetname=2)

        data[x, y:y + 6] = F1_measures.iloc[0]
        data[x + 1, y:y + 6] = F1_measures.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in F1_measures.columns])

F_data_test_day = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 6 * 3))
data[:] = np.nan

f = frequency['5 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 6

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        F1_measures = pd.read_excel(my_path_data, sheetname=2)

        data[x, y:y + 6] = F1_measures.iloc[0]
        data[x + 1, y:y + 6] = F1_measures.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in F1_measures.columns])

F_data_test_5min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 6 * 3))
data[:] = np.nan

f = frequency['3 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 6

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        F1_measures = pd.read_excel(my_path_data, sheetname=2)

        data[x, y:y + 6] = F1_measures.iloc[0]
        data[x + 1, y:y + 6] = F1_measures.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in F1_measures.columns])

F_data_test_3min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 6 * 3))
data[:] = np.nan

f = frequency['1 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 6

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        F1_measures = pd.read_excel(my_path_data, sheetname=2)

        data[x, y:y + 6] = F1_measures.iloc[0]
        data[x + 1, y:y + 6] = F1_measures.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in F1_measures.columns])

F_data_test_min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################

# write = pd.ExcelWriter(my_path_output)
# F_data_test_day.to_excel(write,sheet_name='2-3-day')
# F_data_test_5min.to_excel(write,sheet_name='2-3-5mim')
# F_data_test_3min.to_excel(write,sheet_name='2-3-3mim')
# F_data_test_min.to_excel(write,sheet_name='2-3-mim')
# write.save()

##################################################


# %% 2-3 Scores

my_path_output = './output_final/output_final_3-3.xlsx'

data = np.zeros((10, 7 * 3))
data[:] = np.nan

f = frequency['Day']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 7

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        Scores = pd.read_excel(my_path_data, sheetname=0)

        data[x, y:y + 7] = Scores.iloc[0]
        data[x + 1, y:y + 7] = Scores.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in Scores.columns])

Scores_data_test_day = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 7 * 3))
data[:] = np.nan

f = frequency['5 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 7

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        Scores = pd.read_excel(my_path_data, sheetname=0)

        data[x, y:y + 7] = Scores.iloc[0]
        data[x + 1, y:y + 7] = Scores.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in Scores.columns])

Scores_data_test_5min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 7 * 3))
data[:] = np.nan

f = frequency['3 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 7

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        Scores = pd.read_excel(my_path_data, sheetname=0)

        data[x, y:y + 7] = Scores.iloc[0]
        data[x + 1, y:y + 7] = Scores.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in Scores.columns])

Scores_data_test_3min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################
data = np.zeros((10, 7 * 3))
data[:] = np.nan

f = frequency['1 Min']

for ixi, ix in enumerate(IX):
    '''
    ixi: 0-4
    '''

    for mti, mt in enumerate(['Logistic', 'SimpleNN', 'LSTM']):
        '''
        mti: 0-2
        '''

        x = ixi * 2
        y = mti * 7

        my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + '1' + '.xlsx'

        Scores = pd.read_excel(my_path_data, sheetname=0)

        data[x, y:y + 7] = Scores.iloc[0]
        data[x + 1, y:y + 7] = Scores.iloc[2]

indexes = pd.MultiIndex.from_tuples([(y, h) for y in IX for h in ['Train', 'Test']])
columns = pd.MultiIndex.from_tuples([(z, h) for z in ['Logistic', 'SimpleNN', 'LSTM'] for h in Scores.columns])

Scores_data_test_min = pd.DataFrame(data, index=indexes, columns=columns)

##################################################

# write = pd.ExcelWriter(my_path_output)
# Scores_data_test_day.to_excel(write,sheet_name='2-3-day')
# Scores_data_test_5min.to_excel(write,sheet_name='2-3-5mim')
# Scores_data_test_3min.to_excel(write,sheet_name='2-3-3mim')
# Scores_data_test_min.to_excel(write,sheet_name='2-3-mim')
# write.save()

##################################################

Scores_data_test_day_norm = Scores_data_test_day.copy()
Scores_data_test_5min_norm = Scores_data_test_5min.copy()
Scores_data_test_3min_norm = Scores_data_test_3min.copy()
Scores_data_test_min_norm = Scores_data_test_min.copy()

for score in Scores.columns:
    s = Scores_data_test_day[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]].copy()
    s = (s - np.mean(s.values.flatten())) / np.std(s.values.flatten())
    Scores_data_test_day_norm[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]] = s
for score in Scores.columns:
    s = Scores_data_test_5min[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]].copy()
    s = (s - np.mean(s.values.flatten())) / np.std(s.values.flatten())
    Scores_data_test_5min_norm[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]] = s
for score in Scores.columns:
    s = Scores_data_test_3min[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]].copy()
    s = (s - np.mean(s.values.flatten())) / np.std(s.values.flatten())
    Scores_data_test_3min_norm[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]] = s
for score in Scores.columns:
    s = Scores_data_test_min[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]].copy()
    s = (s - np.mean(s.values.flatten())) / np.std(s.values.flatten())
    Scores_data_test_min_norm[[('LSTM', score), ('SimpleNN', score), ('Logistic', score)]] = s

columns_high = [(x, y) for x in ['Logistic', 'SimpleNN', 'LSTM'] for y in ['Accuracy', 'AUC']]
columns_low = [(x, y) for x in ['Logistic', 'SimpleNN', 'LSTM'] for y in ['Cost', 'mse', 'msle', 'mse', 'mape']]

Scores_data_test_day_high = Scores_data_test_day_norm[columns_high]
Scores_data_test_day_low = Scores_data_test_day_norm[columns_low]

Scores_data_test_5min_high = Scores_data_test_5min_norm[columns_high]
Scores_data_test_5min_low = Scores_data_test_5min_norm[columns_low]

Scores_data_test_3min_high = Scores_data_test_3min_norm[columns_high]
Scores_data_test_3min_low = Scores_data_test_3min_norm[columns_low]

Scores_data_test_min_high = Scores_data_test_min_norm[columns_high]
Scores_data_test_min_low = Scores_data_test_min_norm[columns_low]

# %%


x, y = np.mgrid[0:10, 0:15]
elev = 30
azim = 70

fig = plt.figure(figsize=(20, 15))

ax = plt.subplot(221, projection='3d')

ax.plot_surface(x, y, Scores_data_test_day_low, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 14, 15))
ax.set_yticklabels(('L_c1  ', 'L_c2  ', 'L_c3  ', 'L_c4  ', 'L_c5  ', \
                    'NN_c1    ', 'NN_c2    ', 'NN_c3    ', 'NN_c4    ', 'NN_c5    ', \
                    'LSTM_c1    ', 'LSTM_c2    ', 'LSTM_c3    ', 'LSTM_c4    ', 'LSTM_c5    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Cost Scores', fontsize=12)
ax.set_title('Cost Scores of Daily Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:15]

# fig = plt.figure(figsize=(15,10))

ax = plt.subplot(222, projection='3d')

ax.plot_surface(x, y, Scores_data_test_5min_low, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 14, 15))
ax.set_yticklabels(('L_c1  ', 'L_c2  ', 'L_c3  ', 'L_c4  ', 'L_c5  ', \
                    'NN_c1    ', 'NN_c2    ', 'NN_c3    ', 'NN_c4    ', 'NN_c5    ', \
                    'LSTM_c1    ', 'LSTM_c2    ', 'LSTM_c3    ', 'LSTM_c4    ', 'LSTM_c5    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Cost Scores', fontsize=12)
ax.set_title('Cost Scores of 5-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:15]

# fig = plt.figure(figsize=(15,10))

ax = plt.subplot(223, projection='3d')

ax.plot_surface(x, y, Scores_data_test_3min_low, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 14, 15))
ax.set_yticklabels(('L_c1  ', 'L_c2  ', 'L_c3  ', 'L_c4  ', 'L_c5  ', \
                    'NN_c1    ', 'NN_c2    ', 'NN_c3    ', 'NN_c4    ', 'NN_c5    ', \
                    'LSTM_c1    ', 'LSTM_c2    ', 'LSTM_c3    ', 'LSTM_c4    ', 'LSTM_c5    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Cost Scores', fontsize=12)
ax.set_title('Cost Scores of 3-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:15]

# fig = plt.figure(figsize=(15,10))

ax = plt.subplot(224, projection='3d')

ax.plot_surface(x, y, Scores_data_test_min_low, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 14, 15))
ax.set_yticklabels(('L_c1  ', 'L_c2  ', 'L_c3  ', 'L_c4  ', 'L_c5  ', \
                    'NN_c1    ', 'NN_c2    ', 'NN_c3    ', 'NN_c4    ', 'NN_c5    ', \
                    'LSTM_c1    ', 'LSTM_c2    ', 'LSTM_c3    ', 'LSTM_c4    ', 'LSTM_c5    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=20)
ax.set_zlabel('Standardized Cost Scores', fontsize=12)
ax.set_title('Cost Scores of 1-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.01, wspace=0.00)

fig.show()

# fig.savefig('./images/2-3-alllow.png')


##################################################

# %%

elev = 30
azim = 70

x, y = np.mgrid[0:10, 0:6]

fig = plt.figure(figsize=(20, 15))

ax = plt.subplot(221, projection='3d')

ax.plot_surface(x, y, Scores_data_test_day_high, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 5, 6))
ax.set_yticklabels(('L_a1  ', 'L_a2  ', \
                    'NN_a1    ', 'NN_a2    ', \
                    'LSTM_a1    ', 'LSTM_a2    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Accuracy Scores', fontsize=12)
ax.set_title('Accuracy Scores of Daily Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:6]

# fig = plt.figure(figsize=(20,15))

ax = plt.subplot(222, projection='3d')

ax.plot_surface(x, y, Scores_data_test_5min_high, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 5, 6))
ax.set_yticklabels(('L_a1  ', 'L_a2  ', \
                    'NN_a1    ', 'NN_a2    ', \
                    'LSTM_a1    ', 'LSTM_a2    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Accuracy Scores', fontsize=12)
ax.set_title('Accuracy Scores of 5-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:6]

# fig = plt.figure(figsize=(20,15))

ax = plt.subplot(223, projection='3d')

ax.plot_surface(x, y, Scores_data_test_3min_high, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 5, 6))
ax.set_yticklabels(('L_a1  ', 'L_a2  ', \
                    'NN_a1    ', 'NN_a2    ', \
                    'LSTM_a1    ', 'LSTM_a2    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Accuracy Scores', fontsize=12)
ax.set_title('Accuracy Scores of 3-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

# fig.show()
#
# fig.savefig('./images/2-3-1-minlow.png')


##################################################
# %
x, y = np.mgrid[0:10, 0:6]

# fig = plt.figure(figsize=(20,15))

ax = plt.subplot(224, projection='3d')

ax.plot_surface(x, y, Scores_data_test_min_high, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_yticks(np.linspace(0, 5, 6))
ax.set_yticklabels(('L_a1  ', 'L_a2  ', \
                    'NN_a1    ', 'NN_a2    ', \
                    'LSTM_a1    ', 'LSTM_a2    '))

ax.set_xticks(np.linspace(0, 9, 10))
ax.set_xticklabels(('HS300_i', 'HS300_o', 'ZZ500_i', 'ZZ500_o', 'SH50_i', 'SH50_o', 'IF_i', 'IF_o', 'IC_i', 'IC_o'))

ax.set_xlabel('Indexes/Futures', fontsize=12, labelpad=15)
ax.set_ylabel('Models', fontsize=12, labelpad=15)
ax.set_zlabel('Standardized Accuracy Scores', fontsize=12)
ax.set_title('Accuracy Scores of 1-Min Data', fontsize=15)

ax.view_init(elev=elev, azim=azim)

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.01, wspace=0.00)

fig.show()

# fig.savefig('./images/2-3-allhigh.png')


# %% 统计性描述

from utils import *

frequency = {'Day': 'day', '5 Min': '5min', '3 Min': '3min', '1 Min': 'min'}
IX = ['HS300', 'ZZ500', 'SH50', 'IF', 'IC']

indexes = pd.MultiIndex.from_tuples([(x, y, z) for x in frequency.keys() for y in IX for z in
                                     ['count', 'min', 'median', 'max', 'mean', 'std', 'skew', 'kurt']])
columns = ['Open', 'High', 'Low', 'Close', 'Volume']

indicator = 'basic'
part = 2
basic = True

for i, freq in enumerate(frequency.values()):
    for j, ix in enumerate(IX):

        print(i, j)

        file_name = "./data/" + ix + "_" + freq + "_" + indicator + ("_" + str(part) if freq == 'min' else '') + ".csv"

        D = DataHandler(file_name)
        data = D.Read(basic, pct=1)

        if (i == 0) and (j == 0):
            stat = mystat(data)
        else:
            stat = np.vstack((stat, mystat(data)))

stat = pd.DataFrame(stat, index=indexes, columns=columns)

# my_path_output = './output_final/output_final_stat.xlsx'
# write = pd.ExcelWriter(my_path_output)
# stat.to_excel(write,sheet_name='allstat')
# write.save()

# %% IC acc

################################################## min train

f = 'min'

fig = plt.figure(figsize=(12, 5))
lw = 2

ix = IX[4]

for j, mt in enumerate(['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM']):
    my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + 'de1' + '.xlsx'
    hist = pd.read_excel(my_path_data, sheetname=3)

    ls = ['+-', '-', '--', '-.', ':'][j]

    plt.subplot(1, 2, 1)
    plt.plot(hist['val_loss'], ls, lw=lw, label=mt)
    plt.legend(loc="upper right")
    plt.xlim([0.0, 50.0])
    plt.ylim([0.68, 0.70])
    plt.ylabel('Cost', fontsize=12)
    plt.xlabel('Epochs', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(hist['val_acc'], ls, lw=lw, label=mt)
    plt.legend(loc="lower right")  # loc=1 
    plt.xlim([0.0, 50.0])
    plt.ylim([0.50, 0.60])
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epochs', fontsize=12)

plt.subplots_adjust(top=0.97, bottom=0.05, left=0.04, right=0.97, hspace=0.2, wspace=0.3)
plt.show()

# fig.savefig('./images/2-4.png')

##################################################


# %% Scores


################################################## 

f = 'min'

data = np.zeros((5 * 2, 7))
data[:] = np.nan

ix = IX[4]

for j, mt in enumerate(['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM']):
    my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + 'de1' + '.xlsx'

    Scores = pd.read_excel(my_path_data, sheetname=0)

    data[j * 2, :] = Scores.iloc[0]
    data[j * 2 + 1, :] = Scores.iloc[1]

indexes = pd.MultiIndex.from_tuples(
    [(y, h) for y in ['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM'] for h in ['Train', 'Test']])
columns = Scores.columns

Scores_data_IC = pd.DataFrame(data, index=indexes, columns=columns)

# my_path_output = './output_final/output_final_score.xlsx'
# write = pd.ExcelWriter(my_path_output)
# Scores_data_IC.to_excel(write,sheet_name='ICscore')
# write.save()

##################################################


# %% F


################################################## 

f = 'min'

data = np.zeros((5 * 2, 6))
data[:] = np.nan

ix = IX[4]

for j, mt in enumerate(['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM']):
    my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + 'de1' + '.xlsx'

    F = pd.read_excel(my_path_data, sheetname=2)

    data[j * 2, :] = F.iloc[0]
    data[j * 2 + 1, :] = F.iloc[1]

indexes = pd.MultiIndex.from_tuples(
    [(y, h) for y in ['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM'] for h in ['Train', 'Test']])
columns = F.columns

F_data_IC = pd.DataFrame(data, index=indexes, columns=columns)

# my_path_output = './output_final/output_final_F.xlsx'
# write = pd.ExcelWriter(my_path_output)
# F_data_IC.to_excel(write,sheet_name='ICscore')
# write.save()

##################################################


# %% CM


################################################## 

f = 'min'

data = np.zeros((5 * 2, 4))
data[:] = np.nan

ix = IX[4]

for j, mt in enumerate(['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM']):
    my_path_data = './output_final/my' + f + '_' + mt + '_ns_p2_' + ix + '_' + 'de1' + '.xlsx'

    CM = pd.read_excel(my_path_data, sheetname=1)

    data[j * 2, :] = CM.iloc[0]
    data[j * 2 + 1, :] = CM.iloc[1]

indexes = pd.MultiIndex.from_tuples(
    [(y, h) for y in ['Logistic', 'SimpleNN', 'DNN', 'RNN', 'LSTM'] for h in ['Train', 'Test']])
columns = CM.columns

CM_data_IC = pd.DataFrame(data, index=indexes, columns=columns)

# my_path_output = './output_final/output_final_CM.xlsx'
# write = pd.ExcelWriter(my_path_output)
# CM_data_IC.to_excel(write,sheet_name='ICCM')
# write.save()

##################################################
