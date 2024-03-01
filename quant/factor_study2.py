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
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from utils.utils import TimeMe
if __name__ == '__main__':
    freeze_support()
    cluster = LocalCluster(n_workers=1, threads_per_worker=4 )
    client = Client(cluster,set_as_default=True)
    with TimeMe('Read dask dataframe from parquet'):
        try:
            simdd = dd.read_parquet('c:/temp/simdd.parquet')
        except:
            with TimeMe('Read csv'):
                simdf = pd.read_csv('c:/temp/FACTOR_SIM_RESULTS_01_18_ENRICHED.csv.gz', compression='gzip' )
                simdf.set_index('sym', inplace=True)
                simdf.sort_values(by=['sym','date'],inplace=True)


            #create dask dataframe from simdf and partition it by symbol
            with TimeMe('Create dask dataframe'):
                all_syms = sorted(simdf.index.unique())
                simdd = dd.from_pandas(simdf,npartitions=1)
                ##set up partitions by sym ranges to ensure that any given sym is one partition only
                ##split all_syms into 10 partitions
                NPARTS = 10
                dividers = [ all_syms[i*len(all_syms)//NPARTS] for i in range(NPARTS) ] + [all_syms[-1]]
                simdd = simdd.repartition(divisions=dividers)


                ##simdd = simdd.set_index('sym')
            #WRITE DASK FRAME TO PARQUET
            with TimeMe('Write dask dataframe to parquet'):
                simdd.to_parquet('c:/temp/simdd.parquet',compression='gzip')

    def symbol_simulate(df:pd.DataFrame,senterLong=0,senterShort=0,sexitLong=0,sexitShort=0)->bool:
        df.sort_values(by='date',inplace=True)
        prevRow = None
        results = []
        currentPos = 0
        for row in df.itertuples(index=True):
            if prevRow is not None and prevRow.date >= row.date:
                raise(f'NOT SORTED!!! for symbol {row.Index}{row.date} {prevRow.Index}-{prevRow.date} ')
            if prevRow is not None:
                if currentPos == 0:
                    if prevRow.signal <= senterLong:
                        currentPos = 1
                    elif prevRow.signal >= senterShort:
                        currentPos = -1
                elif currentPos == 1:
                    if prevRow.signal >= sexitLong:
                        currentPos = 0
                elif currentPos == -1:
                    if prevRow.signal <= sexitShort:
                        currentPos = 0
            results.append([row.date,currentPos])
            prevRow = row
        return pd.DataFrame(results,columns=['date','pos']).set_index('date')

    #creage groupby object
    gb = simdd.groupby(simdd.index)
    ENTER_LONG = -1.5
    ENTER_SHORT = 1.5
    EXIT_LONG = 0
    EXIT_SHORT = 0
    #simdd['pos']= 0

    #simdd['pos'] = simdd['pos'].mask((simdd.index == simdd.shift(1).index)&(simdd.shift(1).pos==0)&(simdd.shift(1).signal<=ENTER_LONG),1).compute().reindex()
    #simdd.loc[(simdd.index == simdd.shift(1).index)&(simdd.shift(1).pos==0)&(simdd.shift(1).signal<=ENTER_LONG), 'pos'] = 1

    with TimeMe('Determining Positions'):
        holdings = gb.apply(symbol_simulate,
                       senterLong=ENTER_LONG,senterShort=ENTER_SHORT,sexitLong=EXIT_LONG,sexitShort=EXIT_SHORT,
                            meta={'pos': 'int8'}).compute()
    with TimeMe('Joining Sim and Positions'):
        simdd2 = simdd.reset_index(drop=False).merge(holdings.reset_index(drop=False), how='left', on=['date', 'sym'])\
            .set_index('sym').compute()

    with TimeMe('Sorting'):
        simdd2.sort_values(by=['sym', 'date'], inplace=True)

    with TimeMe('Writing back to '):
        all_syms = sorted(simdd2.index.unique())
        NPARTS = 10
        simdd2 = dd.from_pandas(simdd2, npartitions=1)
        dividers = [all_syms[i * len(all_syms) // NPARTS] for i in range(NPARTS)] + [all_syms[-1]]
        simdd2 = simdd2.repartition(divisions=dividers)

    with TimeMe('Write dask dataframe to parquet'):
        simdd2 = simdd2.set_index('sym')
        simdd2.to_parquet('c:/temp/simdd2.parquet', compression='gzip')
        simdd2 = dd.read_parquet('c:/temp/simdd2.parquet')

    with TimeMe('Extract trades'):
        simdd_trade = simdd2[(simdd2.pos != 0) | (simdd2.shift(1).pos != 0) | (simdd2.shift(-1).pos != 0)].compute()
        simdd_trade['ew_return'] = simdd_trade['pos'] * simdd_trade['excess_c2c']
        simdd_trade = simdd_trade[[c for c in simdd_trade.columns if  not c.startswith('T')]] # remove T0-5 columns
        simdd_trade.to_parquet('c:/temp/simdd_trade_2_1_2024',compression='gzip')
        #drets = simdd_trade.groupby(simdd_trade.date)['ew_return'].mean().compute()
