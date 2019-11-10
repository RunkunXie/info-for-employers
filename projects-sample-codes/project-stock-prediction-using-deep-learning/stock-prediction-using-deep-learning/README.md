# Application of Deep Learning in the Prediction of Stock Trend 

**Summary in one sentence:** 

This project predicted price trends of stock indexes and index futures using Deep Learning models (including ANN, RNN, LSTM), built trading strategies based on various trained models, and found that the LSTM-based high-frequency trading strategy achieved 2.38 sharpe ratio and outcompeted other models.

### 1. Abstract:

Due to the complex and non-linear characteristics of stock prices, the analysis and prediction of stock prices has always been a challenging task in the field of time series analysis. In the field of artificial intelligence, deep learning has demonstrated its strong ability to fit in areas such as image recognition and natural language processing, and has set off an upsurge of research. Among them, the Long Short-Term Memory (LSTM) network not only has the ability of deep learning to fit high-dimensional data, but also has outstanding performance in sequence modeling due to its unique cyclic structure and memory structure. Choosing a suitable deep learning model and applying it to the description and prediction of the complex financial market is undoubtedly of great practical value and theoretical significance.

Utilized deep learning, this paper analyzes and researches the application of deep learning in the Chinese market to forecast the stock price trend. This paper first introduces the theory of deep learning in a relatively comprehensive way, and then discusses the method of applying deep learning to construct a forecast model of stock price trend. Finally, the model is applied to empirical research in the Chinese market. We have found that when using basic market data as input variables, the deep learning model represented by deep LSTM has excellent predictive power for the stock index with 1 minute data, and its prediction accuracy for the CSI 500 index reaches 69.0%; With the increase of data frequency and the increase of sample size, the performance of deep learning model in stock index prediction gradually increased, while the forecasting ability of stock index futures did not change significantly. After introducing technical indicators, the trading strategy based on the deep LSTM model is better than buy and hold strategy and strategies based on other reference models, and achieve 5.58% cumulative returns, 8.23 times annualized returns and a Sharpe ratio of 2.78 without considering transaction costs.

### 2. Project Structure:

**- data**: raw data folders, containing stock price csv files.

**- output:** model results, including model parameters, training logs, evaluation statistics.

**- tables:** tables used the paper.

**- images:** graphs used by the paper. 

### 3. Python Files:

**- main_StockPrediction.py**: 

​	Main Program which does the following:
​	1 Access data, using DataHandler.py
​    2 Make prediction, using models in Model.py
​    3 Evaluate results, using Evaluation.py
​    4 Backtest strategy
​    5 Summarize strategy results

**- DataHandler.py**:
    Read stored csv data, generate training, validation, and test sets.

**- Model.py**:
    Implemented different models, including:
    1 Logistic Regression
    2 Basic Neural Network, ANN
    3 Recurrent Neural Network, RNN
    4 Long Short Term Memory Network, LSTM

**- Evaluation.py**:

​	Evaluate model results.

**- AccessData.py**:
    Read previous results, generate graphs and tables.

**- utils.py**:
    Some support functions.





