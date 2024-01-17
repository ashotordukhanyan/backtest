import pandas as pd

from market_data.market_data import  get_adj_closes, get_daily_stats
from symbology.compositions import get_composition
from utils.utils import get_trading_month_ends,get_trading_days,get_last_trading_month_end,get_previous_trading_day
import logging
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import  date,timedelta
import numpy as np
import functools
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, freeze_support
from sklearn.linear_model import Ridge,Lasso,LinearRegression

from quant.acrch_study import  getCleanData

from market_data.market_data import get_daily_stats
def getStockEtfCleanData(TRAIN_PERIOD_START,TRAIN_PERIOD_END,universe='IWB',etfs=['SPY'],remove_outliers=True):
    tickers = list(get_composition(universe,get_last_trading_month_end(TRAIN_PERIOD_END)).ticker.unique()) if universe else []
    for etf in etfs:
        if etf not in tickers:
            tickers.append(etf)
    return getCleanData(TRAIN_PERIOD_START,TRAIN_PERIOD_END,sorted(tickers),benchmark=None,remove_outliers=remove_outliers)


from utils.utils import cached_df
@cached_df
def simulateOneTradeDate(TRADE_DATE:date,HEDGE_ETFS_PER_NAME = 3,SIG_CUTOFF=1.0, BETA_R_ALPHA=.0001, universe='IWB',codever=0)->pd.DataFrame :
    logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s',datefmt='%H:%M:%S')
    logging.warning(f'TRADE DATE {TRADE_DATE.strftime("%Y-%m-%d")}' )
    TRAIN_PERIOD_START = TRADE_DATE
    TRAIN_PERIOD_END = TRAIN_PERIOD_START +  relativedelta(months=6) - relativedelta(days=1)
    TEST_PERIOD_START = TRAIN_PERIOD_END + timedelta(days=1)
    TEST_PERIOD_END= TEST_PERIOD_START+ relativedelta(months=1) - relativedelta(days=1)

    FACTOR_ETFS = 'SPY RTH XLF XLY XLP SMH XLI XLU XLV KRE IYR OIH XLK IYT XLE QQQ'.split()
    data = getStockEtfCleanData(TRAIN_PERIOD_START, TRAIN_PERIOD_END, universe, FACTOR_ETFS)
    data['date'] = data.date.astype("datetime64[s]")
    daily = data.pivot(index='date',columns='sym',values='c2c')
    corr = daily.corr()

    stocks = sorted(list(set(daily.columns) - set(FACTOR_ETFS)))
    corr = corr[corr.index.isin(FACTOR_ETFS)][stocks] #columns are stocks,rows are factor ETFS
    lrcf = pd.DataFrame(index=stocks, columns=FACTOR_ETFS).fillna(0)
    for stock in stocks:
        hedges = corr[stock].sort_values(ascending=False).head(HEDGE_ETFS_PER_NAME).index #NB no abs() - dont trust neg betas
        rr = Ridge(BETA_R_ALPHA)
        rr.fit(daily[hedges],daily[stock])
        r2 = rr.score(daily[hedges],daily[stock])
        #logging.info(f'{stock} hedges {hedges} R2 {rr.score(daily[hedges],daily[stock])}')
        lrcf.loc[stock, hedges] = rr.coef_


    predictedReturns = daily[FACTOR_ETFS].dot(lrcf.T)
    excessReturns = daily[stocks]-predictedReturns

    from quant.OUSolver import estimate_OU_params,OUParams
    def temp(x):
        ou = estimate_OU_params(x.to_numpy())
        return (ou.alpha,ou.gamma,ou.beta)
    params = excessReturns.apply(temp,axis=0).set_index(pd.Series(['alpha','gamma','beta'])).T
    params['characteristic_time'] = 1/params.alpha
    params['sigma_equilibrium'] = params.beta/np.sqrt(2*params.alpha)


    test_data = getCleanData(get_previous_trading_day(TEST_PERIOD_START,days_back=2),
                             get_previous_trading_day(TEST_PERIOD_END, days_back=-5),
                             sorted(list(set(stocks+FACTOR_ETFS))), remove_outliers=False,benchmark=None)
    test_data['date'] = test_data.date.astype("datetime64[s]")
    ##Add c2c shifted forward by 1-5 days for test_data
    for i in range(0,6):
        test_data[f'c2c_{i}'] = np.nan
        test_data.loc[test_data.sym == test_data.sym.shift(1), f'c2c_{i}'] = test_data.c2c.shift(-1*i)

    test_daily = test_data.pivot(index='date',columns='sym',values='c2c')
    test_stocks = sorted(list(set(test_daily.columns) - set(FACTOR_ETFS)))
    test_excessReturns = test_daily[test_stocks]-test_daily[FACTOR_ETFS].dot(lrcf.loc[test_stocks].T)
    #signal S is calculated as (X[t]-gamma)/sigma_eq
    def calcSignal(X:pd.Series,p:pd.DataFrame)->pd.Series:
        ##X is a series of excess returns
        ##p is pre-calibrated OU params per stock
        sp = p.loc[X.name] #stock params
        return (X-sp.gamma)/sp.sigma_equilibrium

    test_signals = test_excessReturns.apply(calcSignal,axis=0,args=(params,))
    #train_signals = excessReturns.apply(calcSignal,axis=0,args=(params,))

    #create a datafrane with all of the results
    #to start with - signals
    result = test_signals.unstack().reset_index().rename(columns={0:'signal'})
    ## add excess returns
    result = result.merge(test_excessReturns.unstack().reset_index().rename(columns={0:'excess_c2c'}),on=['date','sym'],how='left')
    ##add params
    result = result.merge(params,left_on='sym',right_index=True,how='left')
    ##add actual c2c,volume,closes etc
    result = result.merge(test_data,on=['date','sym'],how='left')
    result['prev_signal']=result.signal.shift(1).fillna(0)
    return result[(result.date.between(TEST_PERIOD_START,TEST_PERIOD_END)) & (result.prev_signal.abs()>=SIG_CUTOFF)]


def run_study():
    freeze_support()
    #include timing in logging
    logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s',datefmt='%H:%M:%S')
    TIME_STEP=relativedelta(months=1)
    TRADE_DATES = []
    TRADE_DATE = date(2007,1,1)
    while ( TRADE_DATE < date(2023,5,1) ):
        TRADE_DATES.append(TRADE_DATE)
        TRADE_DATE  = TRADE_DATE + TIME_STEP

    allSimResults = []
    PARRALEL = 1
    if PARRALEL>1:
        distFunc = functools.partial(simulateOneTradeDate,HEDGE_ETFS_PER_NAME = 1,SIG_CUTOFF=1.0, BETA_R_ALPHA=.0001,codever=20240111)
        with Pool(PARRALEL) as p:
            allSimResults = p.map(distFunc, TRADE_DATES)
            #allSimResults = p.map(simulate, TRADE_DATES)
    else:
        for TRADE_DATE in TRADE_DATES:
            simResults = simulateOneTradeDate(TRADE_DATE,HEDGE_ETFS_PER_NAME = 1,SIG_CUTOFF=1.0, BETA_R_ALPHA=.0001,codever=20240111)
            TRADE_DATE  = TRADE_DATE + TIME_STEP
            allSimResults.append(simResults)

    allSimResults = pd.concat(allSimResults)
    allSimResults.to_csv('c:/temp/FACTOR_SIM_RESULTS_01_11.csv',mode="w+")


if __name__ == '__main__':
    run_study()