from symbology.compositions import get_composition_history
from utils.utils import cached_df
from typing import List
from datetime import date
import yfinance as yf
import logging
import pandas as pd
from functools import lru_cache
from utils.utils import get_trading_days
from utils.qpthack import QConnection
from tqdm import tqdm
from datetime import timedelta

trading_days = sorted(get_trading_days('NYSE',date(2000,1,1),date(2023,9,1)))
trading_days_ix = {td:i for i,td in enumerate(trading_days)}


with QConnection('localhost', 12345, pandas=True) as q:
    p = q.sendSync('select asc distinct date by sym from eodhd_price where year >= 2000,date<=2023.09.01, adjusted_close>0, volume>0')
#p.reset_index(drop=False, inplace=True)
#p['sym'] = p.sym.str.decode('utf-8')
#get a composition of sp500 as of 2001
from symbology.compositions import get_composition_history
r1 = get_composition_history('IVV',date(2000,1,1),date.today())

syms = sorted(p.index.to_list())
res = {}
for sym in tqdm(syms):
    svd = sorted(p.loc[sym]['date'].dt.date) #stock valid days
    periods = [ [svd[0]] ]
    for i in range(1,len(svd)):
        if (trading_days_ix[svd[i]] - trading_days_ix[svd[i-1]] == 1): # no gap
            periods[-1].append(svd[i])
        else:
            #gap !!!
            periods.append([svd[i]])
    MIN_PERIOD_DAYS = 30
    pse = [ (x[0],x[-1]) for x in periods if (x[-1]-x[0])>timedelta(days=MIN_PERIOD_DAYS)] #periods start end
    if pse:
        res[sym.decode()]=pse

    #import matplotlib.pyplot as plt
    #plt.hist([len(x) for x in res.values()],bins=1000)
    #plt.show()

    longest = {k:v for k,v in res.items() if len(v)>=30}