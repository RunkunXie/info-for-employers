# -*- coding: utf-8 -*-

import math
from scipy.stats import norm

e = math.e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Utilities import EuropeanOptions, EW_all
from Input import *


# %%

class RiskCal:
    '''
    Main Member Functions:
        cal_portfolio_price:            Calculate portfolio price
            
        cal_parametric_risk_GBM:        Calculate parametric VaR and ES, assume portfolio follows GBM
        cal_parametric_risk_Normal:     Calculate parametric VaR and ES, assume stocks follows GBM, portfolio follows normal distribution
        cal_historical_risk_relative:   Calculate historical VaR and ES of portfolio using relative return
        cal_historical_risk_absolute:   Calculate historical VaR and ES of portfolio using absolute return
        cal_MonteCarlo_risk_GBM:        Calculate MonteCarlo VaR and ES, assume portfolio follows GBM
        cal_MonteCarlo_risk_Normal:     Calculate MonteCarlo VaR and ES, assume stocks follows GBM, portfolio follows normal distribution
        
        backtest:                       Backtest all VaR to the history, and plot the results
        print_results:                  Print all the results
        
    Other Member Function:
        _cal_option_price:                  Calculate option price of option in the portfolio
        _cal_portfolio_price_no_option:     Calculate portfolio price without option
        _cal_portfolio_price_with_option:   Calculate portfolio price with option

        _cal_logrtn:                        Calculate log return of price series
        _cal_paras_window:                  Calculate GBM parameters using window method
        _cal_paras_EW:                      Calculate GBM parameters using Exponential Weighting method
    '''

    def __init__(self, dates, notional=10000, T=20, t=5, window=5, method='EW', using_option=False,
                 stock_position=None, stock_price=None,
                 option_position=None, option_vol=None, underlying_price=None, maturity=None, option_type=None,
                 risk_free=None):
        """
        Params:
            Basic Setups:
                dates:              np.array, shape=(T*252, 1), datetime in investment period
                notional:           int, total invest amount
                T:                  int, investment period in YEAR
                t:                  int, calculate t-days VaR and ES
                window:             int, window length in yearth
                method:             string, ['EW','Window'], method used to calculate parameters
                using_option:       bool, True if the user want to use option, False otherwise
                
            Stock Parameters:
                stock_position:     np.array, shape=(N_stock, 1), stock positions at the initial date (T years ago)
                stock_price:        np.array, shape=((T+window)*252, N_stock), stock prices start from (T + window) years ago

            Option parameters:            
                option_position:    np.array, shape=(N_option, 1), option positions at the initial date (T years ago)
                underlying_price:   np.array, shape=((T+window)*252, N_option), option underlying prices start from (T + window) years ago            
                option_vol:         np.array, shape=((T+window)*252, N_option), option implied volatility start from (T + window) years ago
                maturity:           np.array, shape=(N_option, 1), time to maturity of the options at the initial date (T years ago)            
                option_type:        np.array, shape=(N_option, 1), option types at the initial date (T years ago)
            
                risk_free:          np.array, shape=((T+window)*252, 1), risk free rates start from (T + window) years ago
                strike:             np.array, shape=((T+window)*252, N_option), option strikes start from (T + window) years ago, default as ATM options
        """

        # Basic Setups
        self.dates = dates[:T * 252]
        self.notional = notional
        self.T = T
        self.t = t
        self.window = window
        self.method = method
        self.using_option = using_option

        # Stock Parameters
        self.stock_position = stock_position
        self.stock_price = stock_price
        self.option_position = option_position
        self.underlying_price = underlying_price
        self.risk_free = risk_free

        # Option parameters
        self.strike = underlying_price
        self.maturity = maturity
        self.option_type = option_type
        self.option_vol = option_vol

        # make sure the position is in line with the assumptions
        if self.option_position is None:
            if np.abs(np.sum(self.stock_position)) - 1 > 1e-3:
                raise ("Total position should be 1 or -1")
        else:
            if np.abs(np.sum(self.stock_position) + np.sum(self.option_position)) - 1 > 1e-3:
                raise ("Total position should be 1 or -1")

    def _cal_option_price(self):
        """
        Calculate option price of option in the portfolio
        
        New Members & Returns:
            option_price:       np.array, shape=((T+window)*252, N_option), option price start from (T + window) years ago
        """

        m, n = len(self.risk_free), len(self.option_position)

        # Duplicate the risk free rates to fit into option pricing
        rf = np.array([self.risk_free for i in range(n)]).reshape(n, m).T
        # Reshape the maturities and option types to fit into option pricing function
        t = self.maturity.reshape(1, n)
        otype = self.option_type.reshape(1, n)

        self.option_price = EuropeanOptions(S=self.underlying_price, K=self.strike, T=t, \
                                            sigma=self.option_vol, r=rf, otype=otype)

        return self.option_price

    def _cal_portfolio_price_no_option(self):
        """
        Calculate portfolio price without option

        New Members & Returns:
            portfolio_price_no_option:      np.array, shape=((T+window)*252, 1), portfolio price without option start from (T + window) years ago
        """

        start_stock_price = self.stock_price[self.T * 252, :]
        start_stock_position_no_option = self.notional * self.stock_position.T / start_stock_price
        self.start_stock_position_no_option = start_stock_position_no_option.round()

        self.portfolio_price_no_option = np.dot(self.stock_price, self.start_stock_position_no_option.T)

        return self.portfolio_price_no_option

    def _cal_portfolio_price_with_option(self):
        """
        Calculate portfolio price with option
        
        New Members & Returns:
            portfolio_price_with_option:      np.array, shape=((T+window)*252, 1), portfolio price with option start from (T + window) years ago
        """

        start_stock_price = self.stock_price[self.T * 252, :]
        start_stock_position_with_option = self.notional * self.stock_position.T / start_stock_price
        self.start_stock_position_with_option = start_stock_position_with_option.round()

        start_option_price = self.option_price[self.T * 252, :]
        start_option_position = self.notional * self.option_position.T / start_option_price
        self.start_option_position = start_option_position.round()

        self.portfolio_price_with_option = np.dot(self.stock_price, self.start_stock_position_with_option.T) + \
                                           np.dot(self.option_price, self.start_option_position.T)

        return self.portfolio_price_with_option

    def cal_portfolio_price(self):
        """
        Calculate portfolio price
        
        New Members & Returns:
            portfolio_price:                np.array, shape=((T+window)*252, 1), portfolio price start from (T + window) years ago
        """

        self._cal_portfolio_price_no_option()

        if self.using_option is True:
            self._cal_option_price()
            self._cal_portfolio_price_with_option()
            self.portfolio_price = self.portfolio_price_with_option
            self.start_stock_position = self.start_stock_position_with_option
        elif self.using_option is False:
            self.portfolio_price = self.portfolio_price_no_option
            self.start_stock_position = self.start_stock_position_no_option

        if (self.portfolio_price > 0).all():
            self.long = True
        elif (self.portfolio_price < 0).all():
            self.long = False
        else:
            raise ("prices series unstable.")

        return self.portfolio_price

    def _cal_logrtn(self, price_series):
        """
        Calculate log return of price series
        
        Params:
            price_series:                  np.array, shape=((T+window)*252, n), portfolio price or portfolio component prices start from (T + window) years ago
        """

        return np.log(price_series[:-1] / price_series[1:])

    def _cal_paras_window(self, price_series):
        """
        Calculate GBM parameters using window method
        
        Params:
            price_series:                  np.array, shape=((T+window)*252, n), portfolio price or portfolio component prices start from (T + window) years ago
            
        Returns:
            mu:                             np.array, shape=((T+window)*252, n), mean/drift of GBM at each date
            sigma:                            np.array, shape=((T+window)*252, n), vol/diffution of GBM at each date
            rho:                             np.array, shape=((T+window)*252, n, n), correlation matrix of GBM at each date
        """

        logrtn = self._cal_logrtn(price_series)

        mu = []
        sigma = []
        rho = []

        for i in range(self.T * 252):
            mu_ba_temp = np.mean(logrtn[i:i + self.window * 252], axis=0)
            sigma_ba_temp = np.std(logrtn[i:i + self.window * 252], axis=0)
            rho_ba_temp = np.corrcoef(logrtn[i:i + self.window * 252], rowvar=0)

            sigma_temp = sigma_ba_temp * (252 ** 0.5)
            mu_temp = mu_ba_temp * 252 + (sigma_temp ** 2) / 2
            rho_temp = rho_ba_temp

            mu.append(mu_temp)
            sigma.append(sigma_temp)
            rho.append(rho_temp)

        mu = np.array(mu)
        sigma = np.array(sigma)
        rho = np.array(rho)

        return mu, sigma, rho

    def _cal_paras_EW(self, price_series, beta=0.9989003714):
        """
        Calculate GBM parameters using Exponential Weighting method
        
        Params:
            price_series:                  np.array, shape=((T+window)*252, n), portfolio price or portfolio component prices start from (T + window) years ago
            beta:                           int, parameters for EW method
            
        Returns:
            mu:                             np.array, shape=((T+window)*252, n), mean/drift of GBM at each date
            sigma:                            np.array, shape=((T+window)*252, n), vol/diffution of GBM at each date
            rho:                             np.array, shape=((T+window)*252, n, n), correlation matrix of GBM at each date
        """

        logrtn = self._cal_logrtn(price_series)

        an = logrtn[::-1]

        # use EW_all to calculate EW terms
        mn, rn = EW_all(an, beta)
        nvars = mn.shape[1]

        # estimate
        mu_ba = np.ones((mn.shape[0], mn.shape[1]))
        sigma_ba = np.ones((mn.shape[0], mn.shape[1]))
        rho_ba = np.ones(rn.shape)

        for i in range(mu_ba.shape[0]):
            mu_ba[i] = mn[i].reshape(-1)
            sigma_ba[i] = (rn[i].diagonal() - mu_ba[i] ** 2) ** 0.5
            rho_ba[i] = (rn[i] - np.dot(mn[i], mn[i].T)) / np.dot(sigma_ba[i].reshape((nvars, 1)),
                                                                  sigma_ba[i].reshape((1, nvars)))

        mu_ba = mu_ba[-self.T * 252:]
        sigma_ba = sigma_ba[-self.T * 252:]
        rho_ba = rho_ba[-self.T * 252:]

        # convert to real parameters
        sigma = sigma_ba * (252 ** 0.5)
        mu = mu_ba * 252 + (sigma ** 2) / 2
        rho = rho_ba

        sigma = sigma[::-1]
        mu = mu[::-1]
        rho = rho[::-1]

        return mu, sigma, rho

    def cal_parametric_risk_GBM(self, pVaR=0.99, pES=0.975):
        '''
        Calculate parametric VaR and ES, assume portfolio follows GBM
        
        Params:
            pVaR:    float, VaR quantile
            pES:     float, ES quantile
        
        New Members:
            parametric_VaR_GBM:     np.array, shape=(T*252, n), parametric VaR
            parametric_ES_GBM:      np.array, shape=(T*252, n), parametric ES
        '''

        t = self.t / 252
        S0 = self.notional

        if self.method is 'EW':
            self.mu_Portfolio, self.sigma_Portfolio, _ = self._cal_paras_EW(self.portfolio_price)
        elif self.method is 'Window':
            self.mu_Portfolio, self.sigma_Portfolio, _ = self._cal_paras_window(self.portfolio_price)

        mu = self.mu_Portfolio
        sigma = self.sigma_Portfolio

        if self.long is True:
            VaR = S0 - S0 * math.e ** (sigma * t ** 0.5 * norm.ppf(1 - pVaR) + (mu - sigma ** 2 / 2) * t)
            ES = S0 * (1 - math.e ** (mu * t) * norm.cdf(norm.ppf(1 - pES) - sigma * t ** 0.5) / (1 - pES))
        elif self.long is False:
            VaR = S0 * math.e ** (sigma * t ** 0.5 * norm.ppf(pVaR) + (mu - sigma ** 2 / 2) * t) - S0
            ES = S0 * (math.e ** (mu * t) * (1 - norm.cdf(norm.ppf(pES) - sigma * t ** 0.5)) / (1 - pES) - 1)

        self.parametric_VaR_GBM = VaR
        self.parametric_ES_GBM = ES

        return None

    def cal_parametric_risk_Normal(self, pVaR=0.99, pES=0.975):
        '''
        Calculate parametric VaR and ES, assume stocks follows GBM, portfolio follows normal distribution
        
        Params:
            pVaR:    float, VaR quantile
            pES:     float, ES quantile
        
        New Members:
            parametric_VaR_Normal:     np.array, shape=(T*252, n), parametric VaR
            parametric_ES_Normal:      np.array, shape=(T*252, n), parametric ES
        '''

        t = self.t / 252
        S0 = self.notional

        if self.method is 'EW':
            self.mu_UnderlyingStock, self.sigma_UnderlyingStock, self.rho_UnderlyingStock = self._cal_paras_EW(
                self.stock_price)
        elif self.method is 'Window':
            self.mu_UnderlyingStock, self.sigma_UnderlyingStock, self.rho_UnderlyingStock = self._cal_paras_window(
                self.stock_price)

        mu = self.mu_UnderlyingStock
        sigma = self.sigma_UnderlyingStock
        rho = self.rho_UnderlyingStock

        V0 = S0 / self.portfolio_price_no_option * self.start_stock_position_no_option * self.stock_price
        V0 = V0[:self.T * 252]

        EVt = np.sum(V0 * e ** (mu * t), axis=1)

        VarVt = []
        for i in range(self.T * 252):
            temp = V0[i] * e ** (mu[i] * t)
            temp = temp.reshape((len(temp), 1))

            temp_sigma = sigma[i]
            temp_sigma = temp_sigma.reshape((len(temp_sigma), 1))

            temp_Cov = np.dot(temp_sigma, temp_sigma.T) * rho[i]

            VarVt_temp = np.dot(np.dot(temp.T, e ** (temp_Cov * t)), temp) - EVt[i] ** 2
            VarVt.append(VarVt_temp[0][0])

        VarVt = np.array(VarVt)
        sdVt = VarVt ** 0.5

        V0 = np.sum(V0, axis=1)

        if self.long is True:
            VaR = V0 - (EVt + norm.ppf(1 - pVaR) * sdVt)
            ES = V0 - (EVt - e ** (-norm.ppf(1 - pES) ** 2 / 2) / ((1 - pES) * (2 * math.pi) ** 0.5) * sdVt)
        elif self.long is False:
            VaR = (EVt + norm.ppf(pVaR) * sdVt) - V0
            ES = (EVt + e ** (-norm.ppf(1 - pES) ** 2 / 2) / ((1 - pES) * (2 * math.pi) ** 0.5) * sdVt) - V0

        self.parametric_VaR_Normal = VaR
        self.parametric_ES_Normal = ES

        return None

    def cal_historical_risk_relative(self, pVaR=0.99, pES=0.975):
        '''
        Calculate historical VaR and ES of portfolio using relative return
        
        Params:
            pVaR:    float, VaR quantile
            pES:     float, ES quantile
        
        New Members:
            historical_VaR_relative:     np.array, shape=(T*252, n), historical VaR
            historical_ES_relative:      np.array, shape=(T*252, n), historical ES
        '''

        price_series = self.portfolio_price
        logrtn = np.log(price_series[:-self.t] / price_series[self.t:])

        VaR = []
        ES = []

        for i in range(self.T * 252):
            rtn_sample = logrtn[i:i + self.window * 252]
            N_sample = self.notional * rtn_sample
            N_sample = N_sample.reshape(-1)

            if self.long is False:
                N_sample = -N_sample

            N_sample.sort()
            VaR.append(- N_sample[int((1 - pVaR) * len(N_sample)) - 1])
            ES.append(- np.mean(N_sample[:int((1 - pES) * len(N_sample))]))

        VaR = np.array(VaR)
        ES = np.array(ES)

        self.historical_VaR_relative = VaR
        self.historical_ES_relative = ES

        return None

    def cal_historical_risk_absolute(self, pVaR=0.99, pES=0.975):
        '''
        Calculate historical VaR and ES of portfolio using absolute return
        
        Params:
            pVaR:    float, VaR quantile
            pES:     float, ES quantile
        
        New Members:
            historical_VaR_relative:     np.array, shape=(T*252, n), historical VaR
            historical_ES_relative:      np.array, shape=(T*252, n), historical ES
        '''

        price_series = self.portfolio_price
        PL = price_series[:-self.t] - price_series[self.t:]

        VaR = []
        ES = []

        for i in range(self.T * 252):
            PL_sample = PL[i:i + self.window * 252]
            N_sample = self.notional * PL_sample / price_series[i]
            N_sample = N_sample.reshape(-1)

            if self.long is False:
                N_sample = -N_sample

            N_sample.sort()
            VaR.append(- N_sample[int((1 - pVaR) * len(N_sample)) - 1])
            ES.append(- np.mean(N_sample[:int((1 - pES) * len(N_sample))]))

        VaR = np.array(VaR)
        ES = np.array(ES)

        self.historical_VaR_absolute = VaR
        self.historical_ES_absolute = ES

        return None

    def cal_MonteCarlo_risk_GBM(self, pVaR=0.99, pES=0.975, sample_size=5000):
        '''
        Calculate MonteCarlo VaR and ES, assume portfolio follows GBM
        
        Params:
            pVaR:                   float, VaR quantile
            pES:                    float, ES quantile
            sample_size:            int, sample size for MC method
        
        New Members:
            MonteCarlo_VaR_GBM:     np.array, shape=(T*252, n), MonteCarlo VaR
            MonteCarlo_ES_GBM:      np.array, shape=(T*252, n), MonteCarlo ES
        '''

        t = self.t / 252
        S0 = self.notional

        mu = self.mu_Portfolio
        sigma = self.sigma_Portfolio

        VaR = []
        ES = []
        for i in range(len(mu)):
            sample = np.random.normal(0, t ** 0.5, sample_size)

            St = S0 * math.e ** ((mu[i] - sigma[i] ** 2 / 2) * t + sigma[i] * sample)
            St.sort()

            if self.long is True:
                VaR.append(S0 - St[int((1 - pVaR) * len(sample)) - 1])
                ES.append(np.mean(S0 - St[:int((1 - pES) * len(sample))]))
            elif self.long is False:
                VaR.append(St[int((pVaR) * len(sample))] - S0)
                ES.append(np.mean(St[int((pES) * len(sample)):] - S0))

        VaR = np.array(VaR)
        ES = np.array(ES)

        self.MonteCarlo_VaR_GBM = VaR
        self.MonteCarlo_ES_GBM = ES

        return None

    def cal_MonteCarlo_risk_Normal(self, pVaR=0.99, pES=0.975, sample_size=5000):
        '''
        Calculate MonteCarlo VaR and ES, assume stocks follows GBM, portfolio follows normal distribution
        
        Params:
            pVaR:                   float, VaR quantile
            pES:                    float, ES quantile
            sample_size:               int, sample size for MC method
        
        New Members:
            MonteCarlo_VaR_GBM:     np.array, shape=(T*252, n), MonteCarlo VaR
            MonteCarlo_ES_GBM:      np.array, shape=(T*252, n), MonteCarlo ES
        '''

        t = self.t / 252
        S0 = self.notional

        mu = self.mu_UnderlyingStock
        sigma = self.sigma_UnderlyingStock
        rho = self.rho_UnderlyingStock

        V0 = S0 / self.portfolio_price_no_option * self.start_stock_position_no_option * self.stock_price
        V0 = V0[:self.T * 252]

        VaR = []
        ES = []
        for i in range(len(mu)):
            sample = np.random.multivariate_normal(np.zeros(mu.shape[1]), rho[i] * t, sample_size)

            Vt = np.sum(V0[i] * math.e ** ((mu[i] - sigma[i] ** 2 / 2) * t + sigma[i] * sample), axis=1)
            Vt.sort()

            if self.long is True:
                VaR.append(S0 - Vt[int((1 - pVaR) * len(sample)) - 1])
                ES.append(np.mean(S0 - Vt[:int((1 - pES) * len(sample))]))
            elif self.long is False:
                VaR.append(Vt[int((pVaR) * len(sample))] - S0)
                ES.append(np.mean(Vt[int((pES) * len(sample)):] - S0))

        VaR = np.array(VaR)
        ES = np.array(ES)

        self.MonteCarlo_VaR_Normal = VaR
        self.MonteCarlo_ES_Normal = ES

        return None

    def backtest(self, pVaR=0.99):
        '''
        Backtest all VaR to the history, and plot the results

        Params:
            pVaR:                   float, VaR quantile        
        '''

        logrtn = np.log(self.portfolio_price[:-self.t] / self.portfolio_price[self.t:])

        if self.long == True:
            portfolio_res = self.notional - self.notional * e ** (logrtn)
        else:
            portfolio_res = self.notional * e ** (logrtn) - self.notional

        if self.using_option == False:
            ntrials = min(len(self.parametric_VaR_GBM), len(self.parametric_VaR_Normal), \
                          len(self.historical_VaR_relative), len(self.historical_VaR_absolute), \
                          len(self.MonteCarlo_VaR_GBM), len(self.MonteCarlo_VaR_Normal))

            Count = pd.DataFrame()

            for i in range(ntrials - 252):
                Count.loc[i, 'parametric_VaR_GBM'] = sum(portfolio_res[i:i + 252] > self.parametric_VaR_GBM[i + 252])
                Count.loc[i, 'parametric_VaR_Normal'] = sum(
                    portfolio_res[i:i + 252] > self.parametric_VaR_Normal[i + 252])
                Count.loc[i, 'historical_VaR_relative'] = sum(
                    portfolio_res[i:i + 252] > self.historical_VaR_relative[i + 252])
                Count.loc[i, 'historical_VaR_absolute'] = sum(
                    portfolio_res[i:i + 252] > self.historical_VaR_absolute[i + 252])
                Count.loc[i, 'MonteCarlo_VaR_GBM'] = sum(portfolio_res[i:i + 252] > self.MonteCarlo_VaR_GBM[i + 252])
                Count.loc[i, 'MonteCarlo_VaR_Normal'] = sum(
                    portfolio_res[i:i + 252] > self.MonteCarlo_VaR_Normal[i + 252])
                Count.loc[i, 'expect'] = 252 * (1 - pVaR)

        else:
            ntrials = min(len(self.parametric_VaR_GBM), len(self.historical_VaR_relative), \
                          len(self.historical_VaR_absolute), len(self.MonteCarlo_VaR_GBM))

            Count = pd.DataFrame()

            for i in range(ntrials - 252):
                Count.loc[i, 'parametric_VaR_GBM'] = sum(portfolio_res[i:i + 252] > self.parametric_VaR_GBM[i + 252])
                Count.loc[i, 'historical_VaR_relative'] = sum(
                    portfolio_res[i:i + 252] > self.historical_VaR_relative[i + 252])
                Count.loc[i, 'historical_VaR_absolute'] = sum(
                    portfolio_res[i:i + 252] > self.historical_VaR_absolute[i + 252])
                Count.loc[i, 'MonteCarlo_VaR_GBM'] = sum(portfolio_res[i:i + 252] > self.MonteCarlo_VaR_GBM[i + 252])
                Count.loc[i, 'expect'] = 252 * (1 - pVaR)

        plt.figure(1)
        for i in range(Count.shape[1]):
            plt.plot(dates[:len(Count)], Count.iloc[:, i], label=Count.columns.values[i])
        plt.legend(loc='upper left')
        plt.title('Backtest: number of exceptions for different VaR calculation')

    def print_results(self):
        '''
        Print all the results
        1. Calculate portfolio price
        2. Calculate various VaR and ES, and plot results in 3 graph (MonteCarlo, Parametric, Historical)
        3. Backtest all VaR to the history, and plot the results

        Params:
            pVaR:                   float, VaR quantile        
        '''

        self.cal_portfolio_price()

        if self.using_option is False:
            self.cal_parametric_risk_GBM()
            self.cal_parametric_risk_Normal()
            self.cal_historical_risk_relative()
            self.cal_historical_risk_absolute()
            self.cal_MonteCarlo_risk_GBM()
            self.cal_MonteCarlo_risk_Normal()

        elif self.using_option is True:
            self.cal_parametric_risk_GBM()
            self.cal_historical_risk_relative()
            self.cal_historical_risk_absolute()
            self.cal_MonteCarlo_risk_GBM()

        if self.using_option is False:
            plt.plot(self.dates, self.MonteCarlo_VaR_GBM, label='MonteCarlo_VaR_GBM')
            plt.plot(self.dates, self.MonteCarlo_ES_GBM, label='MonteCarlo_ES_GBM')

            plt.plot(self.dates, self.MonteCarlo_VaR_Normal, label='MonteCarlo_VaR_Normal')
            plt.plot(self.dates, self.MonteCarlo_ES_Normal, label='MonteCarlo_ES_Normal')

            plt.title("MonteCarlo VaR and ES")
            plt.legend()
            plt.show()

            plt.plot(self.dates, self.parametric_VaR_GBM, label='parametric_VaR_GBM')
            plt.plot(self.dates, self.parametric_ES_GBM, label='parametric_ES_GBM')

            plt.plot(self.dates, self.parametric_VaR_Normal, label='parametric_VaR_Normal')
            plt.plot(self.dates, self.parametric_ES_Normal, label='parametric_ES_Normal')

            plt.title("Parametric VaR and ES")
            plt.legend()
            plt.show()

            plt.plot(self.dates, self.historical_VaR_relative, label='historical_VaR_relative')
            plt.plot(self.dates, self.historical_ES_relative, label='historical_ES_relative')

            plt.plot(self.dates, self.historical_VaR_absolute, label='historical_VaR_absolute')
            plt.plot(self.dates, self.historical_ES_absolute, label='historical_ES_absolute')

            plt.title("Historical VaR and ES")
            plt.legend()
            plt.show()

        elif self.using_option is True:
            plt.plot(self.dates, self.MonteCarlo_VaR_GBM, label='MonteCarlo_VaR_GBM')
            plt.plot(self.dates, self.MonteCarlo_ES_GBM, label='MonteCarlo_ES_GBM')

            plt.title("MonteCarlo VaR and ES")
            plt.legend()
            plt.show()

            plt.plot(self.dates, self.parametric_VaR_GBM, label='parametric_VaR_GBM')
            plt.plot(self.dates, self.parametric_ES_GBM, label='parametric_ES_GBM')

            plt.title("Parametric VaR and ES")
            plt.legend()
            plt.show()

            plt.plot(self.dates, self.historical_VaR_relative, label='historical_VaR_relative')
            plt.plot(self.dates, self.historical_ES_relative, label='historical_ES_relative')

            plt.plot(self.dates, self.historical_VaR_absolute, label='historical_VaR_absolute')
            plt.plot(self.dates, self.historical_ES_absolute, label='historical_ES_absolute')

            plt.title("Historical VaR and ES")
            plt.legend()
            plt.show()

        self.backtest()


# %% Stock Test Case 1, long 2 stocks, EW

stock_price, stock_position, dates, risk_free = StocksRead().get_data()

'''
If no access to SQLite database, then use the following code to import the data from csv files.

stock_price = np.array(pd.read_csv('data/stock_price.csv', header = 0))
stock_position = np.array(pd.read_csv('data/stock.csv', header = 0).loc[:, 'Position']).reshape(3,1)
d = pd.read_csv('data/dates.csv', header = 0)
d['dates'] = pd.to_datetime(d['dates'])
dates = np.array(d)
risk_free = np.array(pd.read_csv('data/rf.csv', header = 0))
'''

'''
stock_position: 0.5 0.5
'''

Risk = RiskCal(dates, T=20, using_option=False,
               stock_position=stock_position,
               stock_price=stock_price)

Risk.print_results()

# %% Stock Test Case 2, short 2 stocks, Window

# user can change stock position by directly changing "stock_position" if the
# user already read the stock data,  or by changing "stock.csv" 
stock_position = np.array([-0.51, -0.51, 0.02]).reshape((3, 1))

'''
stock_position: -0.51 -0.51 +0.02
'''

Risk = RiskCal(dates, T=20, using_option=False, method='Window', \
               stock_position=stock_position, \
               stock_price=stock_price)

Risk.print_results()

# %% Option Test Case 1, long 3 stocks, long 2 put options, EW

# user can change stock and option position by directly changing "stock_position" 
# and "option_position" if the user already read the stock data, or changing
# "stock.csv" and "option.csv"
stock_position[:] = 0.32

'''
Use following code if no access to SQLite:
    
underlying_price = np.array(pd.read_csv('data/underlying.csv', header = 0))
option_position = np.array(pd.read_csv('data/option.csv', header = 0).loc[:, 'Position']).reshape(2,1)
option_type = np.array(pd.read_csv('data/option.csv', header = 0).loc[:, 'Type'])
maturity = np.array(pd.read_csv('data/option.csv', header = 0).loc[:, 'Maturity'])
option_vol = np.array(pd.read_csv('data/iv.csv', header = 0).iloc[:, 1:])/100
'''

underlying_price, option_position, option_type, maturity, option_vol = OptionsRead().get_data()

option_position[:] = 0.02

stock_price, dates, risk_free, underlying_price, option_vol = \
    ProcessData(stock_price, dates, risk_free, underlying_price, option_vol).process()

'''
stock_position: 0.32 0.32 0.32
option_position: 0.02 0.02
'''

Risk = RiskCal(dates, T=8, using_option=True,
               stock_position=stock_position,
               stock_price=stock_price, option_position=option_position,
               underlying_price=underlying_price, maturity=maturity,
               risk_free=risk_free, option_type=option_type, option_vol=option_vol)

Risk.print_results()

# %% Option Test Case 1, short 3 stocks, long 2 call options, Window

stock_position[:] = -0.34
option_position[:] = 0.01
option_type[:] = 1

'''
stock_position: -0.34 -0.34 -0.34
option_position: 0.01 0.01
'''

Risk = RiskCal(dates, T=8, using_option=True,
               stock_position=stock_position,
               stock_price=stock_price, option_position=option_position,
               underlying_price=underlying_price, maturity=maturity,
               risk_free=risk_free, option_type=option_type, option_vol=option_vol)

Risk.print_results()
