from market_data.extractalpha.datasets import EAEstimize
from datetime import date
from strategy.ensemble import filterByUniverse
from quant.symreturns import SymReturns
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
START_DATE,END_DATE = date(2010,1,1),date(2023,12,31)
eas = EAEstimize()

data = eas.retrieveSignal(START_DATE,END_DATE, additionalWhereClause = '((signal>50)|(signal<-50))')
data.rename(columns={'ticker':'sym','Date':'date'},inplace=True)
data = filterByUniverse(data,universe='IWV')
sr = SymReturns()
data = sr.enrichWithReturns(data,columns=['sym', 'volume', 'date', 'o2c', 'o2cdn', 'o2csn', 'o2cbn'])
minVolume, minPrice = 10000.0, 4.0
data = data[(data.volume > minVolume) & (data.close >= minPrice)]
data = data.replace([np.inf, -np.inf], np.nan).dropna()