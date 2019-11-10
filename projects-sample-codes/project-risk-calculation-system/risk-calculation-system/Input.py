# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:35:06 2018

Read the portfolio information from stocks.csv, option.csv, and iv.csv.
Extract stock prices and risk-free rates from SQLite database.
Process the data to make them have equal length.

@author: Administrator
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import sqlite3
import datetime


class StocksRead:
    # define a class to read the stock positions and get stock data

    def __init__(self):
        self.stocks = pd.read_csv('data/stock.csv', header=0)
        self.tickers = list(self.stocks.loc[:, 'Stock'])
        # Eg. [XOM, INTC]
        self.pos = list(self.stocks.loc[:, 'Position'])
        # EG. [0.5, 0.5]

    def obtain(self, ticker):
        # determine whether we have the data for this ticker

        conn = sqlite3.connect('data/stocks.sqlite')
        # connect to the database stored in the file stocks.sqlite
        cur = conn.cursor()
        # create a handle to perform operations on the data

        with conn:
            cur.execute('select id from symbol where ticker = ?', (ticker,))
            # look for the ticker in the table symbol           
            row = cur.fetchone()
        cur.close()

        return 1 if row else 0

    def merge_data(self, df, stock_id, ticker):
        # merge the data for all tickers together into a DataFrame

        conn = sqlite3.connect('data/stocks.sqlite')
        cur = conn.cursor()

        if df.shape[1] == 0:
            # If the DataFrame is empty, then we need to set the index column, which contains the dates
            df = pd.read_sql_query('''
                select price_date, adj_close_price from price where symbol_id = %s
                ''' % str(stock_id), conn)
            # extract the dates and prices from the table price
            df['price_date'] = pd.to_datetime(df['price_date'])
            df.set_index('price_date', inplace=True)
            # set the dates as index column
            df.columns = [ticker]

        else:
            # If there is already data, then merge the new data into it
            new = pd.read_sql_query('''
                select price_date, adj_close_price from price where symbol_id = %s
                ''' % str(stock_id), conn)
            new['price_date'] = pd.to_datetime(new['price_date'])
            new.set_index('price_date', inplace=True)
            new.columns = [ticker]

            df = pd.concat([df, new], axis=1, join='inner')
            # merge the DataFrame new to the existing DataFrame

        cur.close()

        return df

    def get_data(self):
        # return all the stock prices, the stock positions, the dates and risk free rates

        conn = sqlite3.connect('data/stocks.sqlite')
        cur = conn.cursor()

        column_str = 'symbol_id, price_date, adj_close_price'
        insert_str = ('?, ' * 3)[:-2]
        final_str = 'insert into price (%s) values (%s)' % (column_str, insert_str)
        # construct the command that will be executed in SQL

        df = pd.DataFrame()

        for ticker in self.tickers:
            if self.obtain(ticker) == 0:
                # If we don't have the data for this ticker, then we need to download it
                with conn:
                    cur.execute('insert into symbol (ticker) values (?)', (ticker,))
                    cur.execute('select id from symbol where ticker = ?', (ticker,))
                    stock_id = cur.fetchone()[0]
                    # insert this ticker into symbol and get its id

                try:
                    rawdata = data.DataReader(ticker, 'yahoo',
                                              datetime.datetime(1985, 1, 1),
                                              datetime.datetime(2018, 11, 23))
                    # download the data from Yahoo Finance

                    price = []
                    for j in range(len(rawdata) - 1, -1, -1):
                        price.append(
                            (stock_id, str(rawdata.index.values[j])[:10],
                             rawdata.iloc[j, 4]
                             )
                        )
                    # put the data into a list of tuples

                    with conn:
                        cur.executemany(final_str, price)
                        # store the data into database

                except:
                    print('Could not download Yahoo data: %s' % ticker)

                df = self.merge_data(df, stock_id, ticker)
                # merge the DataFrame

            else:
                # If we already have the data, then we just need to extract it from database
                cur.execute('select id from symbol where ticker = ?', (ticker,))
                stock_id = cur.fetchone()[0]

                df = self.merge_data(df, stock_id, ticker)

        # delete all dates that have missing values
        df = df.dropna(axis=0)

        # get the dates
        d = np.array(pd.DataFrame(df.index))

        # Extract the risk free rates from the database
        rf = np.zeros(len(df))
        for i in range(len(rf)):
            cur.execute('select risk_free_rate from rf where rf_date = ?', (str(d[i][0])[:10],))
            rf[i] = cur.fetchone()[0]

        cur.close()

        return np.array(df), np.array(self.pos).reshape(len(self.pos), 1), d, rf.reshape(len(rf), 1)


class OptionsRead:
    # define a class to read the option positions and get underlying stock data

    def __init__(self):
        self.options = pd.read_csv('data/option.csv', header=0)
        self.tickers = list(self.options.loc[:, 'Ticker'])
        self.pos = list(self.options.loc[:, 'Position'])
        # EG. [0.01, 0.01]
        self.type = list(self.options.loc[:, 'Type'])
        # EG. [1, 0]. 1 is call and 0 is put.
        self.t = list(self.options.loc[:, 'Maturity'])
        # EG. [1, 2]. Maturity is in year.
        self.vol = pd.read_csv('iv.csv', header=0).iloc[:, 1:]

    def obtain(self, ticker):
        # determine whether we have the data for this ticker

        conn = sqlite3.connect('data/stocks.sqlite')
        # connect to the database stored in the file stocks.sqlite
        cur = conn.cursor()
        # create a handle to perform operations on the data

        with conn:
            cur.execute('select id from symbol where ticker = ?', (ticker,))
            # look for the ticker in the table symbol           
            row = cur.fetchone()
        cur.close()

        return 1 if row else 0

    def merge_data(self, df, option_id, ticker):
        # merge the data for all tickers together into a DataFrame

        conn = sqlite3.connect('data/stocks.sqlite')
        cur = conn.cursor()

        if df.shape[1] == 0:
            # If the DataFrame is empty, then we need to set the index column, which contains the dates
            df = pd.read_sql_query('''
                select price_date, adj_close_price from price where symbol_id = %s
                ''' % str(option_id), conn)
            # extract the dates and prices from the table price
            df['price_date'] = pd.to_datetime(df['price_date'])
            df.set_index('price_date', inplace=True)
            # set the dates as index column
            df.columns = [ticker]

        else:
            # If there is already data, then merge the new data into it
            new = pd.read_sql_query('''
                select price_date, adj_close_price from price where symbol_id = %s
                ''' % str(option_id), conn)
            new['price_date'] = pd.to_datetime(new['price_date'])
            new.set_index('price_date', inplace=True)
            new.columns = [ticker]

            df = pd.concat([df, new], axis=1, join='inner')
            # merge the DataFrame new to the existing DataFrame

        cur.close()

        return df

    def get_data(self):
        # return all the underlying stock prices, the option positions, the option types
        # (call or put), the maturities and the implied volatilities

        conn = sqlite3.connect('data/stocks.sqlite')
        cur = conn.cursor()

        column_str = 'symbol_id, price_date, adj_close_price'
        insert_str = ('?, ' * 3)[:-2]
        final_str = 'insert into price (%s) values (%s)' % (column_str, insert_str)
        # construct the command that will be executed in SQL

        df = pd.DataFrame()

        for ticker in self.tickers:
            if self.obtain(ticker) == 0:
                # If we don't have the data for this ticker, then we need to download it
                with conn:
                    cur.execute('insert into symbol (ticker) values (?)', (ticker,))
                    cur.execute('select id from symbol where ticker = ?', (ticker,))
                    option_id = cur.fetchone()[0]
                    # insert this ticker into symbol and get its id

                try:
                    rawdata = data.DataReader(ticker, 'yahoo',
                                              datetime.datetime(1985, 1, 1),
                                              datetime.datetime(2018, 11, 23))
                    # download the data from Yahoo Finance

                    price = []
                    for j in range(len(rawdata) - 1, -1, -1):
                        price.append(
                            (option_id, str(rawdata.index.values[j])[:10],
                             rawdata.iloc[j, 4]
                             )
                        )
                    # put the data into a list of tuples

                    with conn:
                        cur.executemany(final_str, price)
                        # store the data into database

                except:
                    print('Could not download Yahoo data: %s' % ticker)

                df = self.merge_data(df, option_id, ticker)
                # merge the DataFrame

            else:
                # If we already have the data, then we just need to extract it from database
                cur.execute('select id from symbol where ticker = ?', (ticker,))
                option_id = cur.fetchone()[0]

                df = self.merge_data(df, option_id, ticker)

        # delete all dates that have missing values
        df = df.dropna(axis=0)

        cur.close()

        return np.array(df), np.array(self.pos).reshape(len(self.pos), 1), \
               np.array(self.type), np.array(self.t), np.array(self.vol) / 100


class ProcessData:
    # Process the data to make them have equal length.

    def __init__(self, stock, dates, rf, underlying, vol):
        self.stock = stock
        self.dates = dates
        self.rf = rf
        self.underlying = underlying
        self.vol = vol

    def process(self):
        T = min(len(self.stock), len(self.underlying), len(self.vol))
        return self.stock[:T], self.dates[:T], self.rf[:T], self.underlying[:T], self.vol[:T]


stock_price, stock_position, dates, risk_free = StocksRead().get_data()
underlying_price, option_position, option_type, maturity, option_vol = OptionsRead().get_data()
stock_price, dates, risk_free, underlying_price, option_vol = \
    ProcessData(stock_price, dates, risk_free, underlying_price, option_vol).process()
