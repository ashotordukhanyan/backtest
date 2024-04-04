import logging
from utils.qpthack import qconnection, qtemporal, MetaData

from market_data.alexandria.alexandria import AlexandriaNewsDailySummary,AlexandriaNewsSentiment
from datetime import date,timedelta
from quant.symreturns import SymReturns
import pandas as pd
from market_data.market_data import get_md_conn
import numpy as np
logging.basicConfig(filename=None,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S')

SD = date(2007,1,1)
ED = date(2023,12,31)
ANDS = AlexandriaNewsDailySummary()
with get_md_conn() as q:
    daily = ANDS.get_sentiments_by_effective_date(SD,ED, q, 'ALL')
daily.drop(columns=['MarketImpactScore','Prob_POS','Prob_NTR','Prob_NEG'],inplace=True)
daily.rename(columns={'EffectiveDate':'date','Ticker':'sym'},inplace=True)
daily['date'] = daily.date.apply(lambda x: x + timedelta(days=1)) ## date represents the date of the news. We are interested in "next days" o2c returns

daily2=SymReturns().enrichWithReturns(daily,['date','sym','o2c','o2cdn','o2cbn','o2csn'])
daily2.dropna(inplace=True)


RELEVANCE_CUTOFF = .5
CONFIDENCE_CUTOFF = .333

df = daily2[(daily2.Relevance>=RELEVANCE_CUTOFF) & (daily2.Confidence>=CONFIDENCE_CUTOFF)]
sentimentCutoffs = [[-1.1,0,0.3,1.1]]
for sc in sentimentCutoffs:
    with pd.option_context('display.float_format', '{:,.5f}'.format):
        stats=df.groupby(pd.cut(df.Sentiment,sc)).agg({'o2cbn':['mean','std','count']})
        stats.columns=['mean','std','count']
        stats['stde']=stats['std']/(stats['count']**.5)
        print(stats)

