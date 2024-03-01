import datetime

from utils.qpthack import qconnection, qtemporal, MetaData
import pandas as pd
from market_data.alexandria.alexandria import AlexandriaNewsDailySummary
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
import logging
from quant.acrch_study import  getCleanData
import matplotlib.pyplot as plt

logging.basicConfig(filename=None,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S')


def get_ls_daily_rets(PERIOD_START:datetime.date,PERIOD_END:datetime.date):
    ''' Returns of a simple strategy where we go long/short sentiment dollar neutral '''
    ands = AlexandriaNewsDailySummary()

    with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
        sentiments = ands.get_all_sentiments(PERIOD_START,PERIOD_END,q)
        tickers = sorted(sentiments.Ticker.unique().tolist())
    sentiment_cols = ['date', 'Ticker', 'Mentions', 'Sentiment', 'Confidence', 'Relevance','MarketImpactScore', 'Prob_POS', 'Prob_NTR', 'Prob_NEG']
    #sentiment_cols except date and Ticker

    sentiments = sentiments[sentiment_cols]
    #enrich with market data
    market_data = getCleanData(PERIOD_START,PERIOD_END,tickers,benchmark='SPY',remove_outliers=True)
    #tempporary until all kdb data conversions are done through datagrid
    market_data['sym'] = market_data.sym.astype('string')


    merged = pd.merge_asof(market_data.sort_values('date'),sentiments,left_by = 'sym', right_by='Ticker',on='date',
                           direction='backward', allow_exact_matches=False,tolerance=pd.Timedelta('3 days'))
    merged = merged.fillna({x:0 for x in sentiments.columns[2:]}).drop(columns=['Ticker'])
    gb = merged.groupby('date')
    #get group keys from gb
    #group = gb.get_group(list(gb.groups.keys())[42])
    def lstrade(group):
        return group[group.Sentiment>0].c2cdn.mean() -group[group.Sentiment<0].c2cdn.mean()
    lsRetsByDate = gb.apply(lstrade)
    details = merged[merged.Sentiment!=0][['date','sym','Sentiment','c2cdn','c2c']]
    return lsRetsByDate,details

#PERIOD_START = datetime.date(2023,1,1)
#PERIOD_END = datetime.date(2023,12,31)

all_daily_rets = []
all_details = []
for year in range(2000,2024):
    logging.info('Processing year %s',year)
    PERIOD_START = datetime.date(year,1,1)
    PERIOD_END = datetime.date(year,12,31)
    daily_rets,detais = get_ls_daily_rets(PERIOD_START,PERIOD_END)
    all_daily_rets.append(daily_rets)
    all_details.append(detais)

all_daily_rets = pd.concat(all_daily_rets)
all_details = pd.concat(all_details)
#(1+lsRetsByDate).cumprod().plot()
#plt.show(block=True)