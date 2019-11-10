**This project develops a risk calculation system.** 

The system is able to:

1. Take a portfolio of stock and option positions as inputs.
2. Both calibrate to historical data or take parameters as inputs.
3. Compute Monte Carlo, historical, and parametric VaR.
4. Compute Monte Carlo, and historical Expected Shortfall.
5. Backtest the computed VaR against history.

***Notes: For more information, see Documentation.***



**The project contains the following files:**

##### 1. Python Scripts:

​	**main_RiskCal** (Main Program)
​	**Input.py**: read the portfolio information from the input files, extracts stock prices and risk-free rates from the SQLite database, and process the data. 

​	**database.py**: generate *stocks.sqlite*, the data file that contains stocks price, interest rate, etc. 

​	**Utilities.py:** define two functions for use in *RiskCal.py*.

##### 2. Data

**2.1Data Files:**
	**stocks.sqlite**: created by file database.py, contains three tables related to each other to streamline the data download and input process.
	**iv.py**: users should give the implied volatility of all the underlying stocks.

**2.2 Setup Files:**
	**stock.csv**: users should specify what stocks they want to long/short and the corresponding positions. 
	**option.csv**: users should specify the underlying stock in column ‘Ticker’, the position for each option in column ‘Position’, the type of each option in column ‘Type’, and the time to maturity for each option in column ‘Maturity’. 

**2.3 Backup Files:** In case some users have no access to SQLite database on their computer, we also prepare 4 csv files with the necessary data for our test cases. They are **stock_price.csv, underlying.csv, dates.csv, and rf.csv**.

