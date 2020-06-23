# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:20:34 2018

Create a SQLite database to store the daily stock prices over the past 30 years.
Download 13-week treasury bill discount rate and construct risk-free rate.

@author: Administrator
"""

from pandas_datareader import data
import sqlite3
import datetime


def create_tables():
    '''
    Create 3 tables in SQLite
    'symbol' shows all the tickers we have. 
    'price' contains the dates and adjusted closing prices of the stocks.
    'rf' contains the dates and risk free rates.
    '''
    # connect to the database stored in the file stocks.sqlite
    conn = sqlite3.connect('data/stocks.sqlite')
    # create a handle to perform operations on the data
    cur = conn.cursor()

    # create 3 tables
    cur.executescript('''
            drop table if exists symbol;
            drop table if exists price;
            drop table if exists rf;

            create table symbol(
            	id integer not null primary key autoincrement unique,
            	ticker varchar(32) not null            	
            );
            
            create table price(            	
            	symbol_id int not null,
            	price_date datetime not null,
            	adj_close_price decimal(19, 4) null,
             imp_vol decimal(19, 4) null
            );
            
            create table rf(
            	rf_date datetime not null,
            	risk_free_rate decimal(19, 4) null       	
            );   
    ''')

    cur.close()


def risk_free_download():
    # We use the 13-week treasury bill discount rate to construct the risk free interest rate.
    # The ^IRX data are the annualized discount rates for the 3-month treasury bill.

    conn = sqlite3.connect('data/stocks.sqlite')
    cur = conn.cursor()

    # download data from Yahoo Finance    
    rawdata = data.DataReader('^IRX', 'yahoo',
                              datetime.datetime(1985, 1, 1),
                              datetime.datetime(2018, 11, 23))

    # Transform the data into annualized risk free rates
    df = (100 / (100 - rawdata.iloc[:, 4] / 4)) ** 4 - 1

    # construct the SQL insert command     
    column_str = 'rf_date, risk_free_rate'
    insert_str = ('?, ' * 2)[:-2]
    final_str = 'insert into rf (%s) values (%s)' % (column_str, insert_str)

    # construct a list tuples containing the elements that need to be inserted
    rf = []
    for j in range(len(df) - 1, -1, -1):
        rf.append(
            (str(df.index.values[j])[:10],
             df.iloc[j,]
             )
        )

    # store the risk free rates into the database
    with conn:
        cur.executemany(final_str, rf)

    cur.close()


create_tables()
risk_free_download()
