from dataclasses import dataclass
import pandas as pd
from utils.utils import get_trading_days,get_last_trading_month_end
import logging
from datetime import  date
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict
from dateutil.relativedelta import relativedelta
from multiprocessing import freeze_support
from market_data.market_data import _kdbdt,get_md_conn
from utils.datagrid import CT, ColumnDef
import warnings
from utils.utils import cached_df
from strategy.base import Trader,TraderParams,TradingSignal
from quant.OUSolver import estimate_OU_params
from utils.utils import get_previous_trading_day


@dataclass
class AvelanedaTraderParams(TraderParams):
    pass

DEFAULT_PARAMS = AvelanedaTraderParams(target = 'c2csn', universe = 'IWV')

class AvelanedaSignal(TradingSignal):
    _SCHEMA = [
        ColumnDef('sym',CT.SYMBOL, isKey=True),
        ColumnDef('date', CT.DATE, isKey=True),
        ColumnDef('signal', CT.F32), ## Signal is based on previous days c2c[*] return and OU parameters calibrated monthly
        ColumnDef('oualpha', CT.F32), # OU process alpha - strength of mean reversion
        ColumnDef('ougamma', CT.F32), # OU process gamma - drift
        ColumnDef('oubeta', CT.F32), # OU process beta - volatility
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(date.today())]*len(frame)),
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.date.dt.year, isPartition=True),
    ]
    def __init__(self):
        super().__init__('avel_signal', self._SCHEMA)
class AvelanedaTrader(Trader):
    ''' Auto-correlation based strategy '''
    def __init__(self,params : AvelanedaTraderParams = DEFAULT_PARAMS):
        self.params_ = params


    def calibrateModels(self,startDate:date,endDate:date) -> pd.DataFrame:
        '''
            Calibrate models for all stocks in universe for given period
            Returns pandas dataframe with OU parameters for each stock's residual return
        '''
        returns = self.getCleanData(startDate,endDate)
        if returns is None or not len(returns):
            return None
        daily = returns.pivot(index='date',columns='sym',values=self.params_.target)

        def temp(x):
            ou = estimate_OU_params(x.to_numpy())
            return (ou.alpha, ou.gamma, ou.beta)

        params = daily.apply(temp, axis=0).set_index(pd.Series(['oualpha', 'ougamma', 'oubeta'])).T
        params['characteristic_time'] = 1 / params.oualpha
        params['sigma_equilibrium'] = params.oubeta / np.sqrt(2 * params.oualpha)
        return params

    def evaluateModels(self, params: pd.DataFrame, startDate:date, endDate:date) -> pd.DataFrame:
        syms = sorted(list(set(params.index.to_list())))
        ##Previous days c2c[*] return is used to calculate todays signal
        returns = self.getCleanData(get_previous_trading_day(startDate),endDate,symbols=syms,removeOutliers=False)
        returns.sort_values(['sym', 'date'], inplace=True)
        returns = returns.merge(params, left_on='sym', right_index=True, how='left')

        returns.loc[returns.sym == returns.sym.shift(1), 'signal'] = (returns[self.params_.target].shift(
            1) - returns.ougamma) / returns.sigma_equilibrium
        return returns[(returns.date >= _kdbdt(startDate)) & (returns.date <= _kdbdt(endDate))][['sym','date','signal','oualpha','ougamma','oubeta']]

@cached_df
def at_simulate(TRAIN_PERIOD_START,TRAIN_PERIOD_END,TEST_PERIOD_START,TEST_PERIOD_END, codever=0) -> pd.DataFrame:
    logging.info(
        f'{index}/{len(SIM_DATES)} TRAINING {TRAIN_PERIOD_START} - {TRAIN_PERIOD_END} TESTING {TEST_PERIOD_START} - {TEST_PERIOD_END}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = trader.calibrateModels(TRAIN_PERIOD_START, TRAIN_PERIOD_END)
    if models is None or not len(models):
        return None
    simResults = trader.evaluateModels(models, TEST_PERIOD_START, TEST_PERIOD_END)
    simResults['asof_date'] = np.datetime64(date.today())
    simResults['year'] = simResults.date.dt.year
    return simResults

if __name__ == '__main__':
    logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    DRY_RUN = False
    freeze_support()
    params = DEFAULT_PARAMS
    SIM_DATES = []
    SIM_DATE = date(2000,7,1)
    while ( SIM_DATE < date(2023,12,1) ):
        SIM_DATES.append(SIM_DATE)
        SIM_DATE  = SIM_DATE + params.trainFrequency

    trader = AvelanedaTrader(params)
    allSimResults = []
    currentProcessedYear = -1
    KDB_ROOT = 'C://KDB_MARKET_DATA2/'
    signal = AvelanedaSignal()
    if not DRY_RUN:
        with get_md_conn() as q:
            signal.kdbInitConnection(q)
    for index,SIM_DATE in enumerate(SIM_DATES):
        TRAIN_PERIOD_START = SIM_DATE - params.trainPeriod
        TRAIN_PERIOD_END = SIM_DATE - relativedelta(days=1)
        TEST_PERIOD_START = SIM_DATE
        TEST_PERIOD_END = SIM_DATE + params.trainFrequency# - relativedelta(days=1)

        simResults = at_simulate(TRAIN_PERIOD_START, TRAIN_PERIOD_END, TEST_PERIOD_START, TEST_PERIOD_END,1)
        if simResults is None or not len(simResults):
            continue
        years = sorted(list(simResults.year.unique()))

        if not DRY_RUN:
            with get_md_conn() as q:
                for year in years:
                    if year > currentProcessedYear:
                        if currentProcessedYear > 0:
                            logging.info(f'Saving data to disk for year {currentProcessedYear}')
                            signal.saveKdbTableToDisk(q, currentProcessedYear, KDB_ROOT)
                        currentProcessedYear = year
                        logging.info(f'Initializing partition table for year {year}')
                        signal.kdbInitPartitionTable(year,q)

                    data = simResults[simResults.year == year].copy()
                    data.meta = signal.getqpythonMetaData()
                    signal.upsertToKDB(q,KDB_ROOT,data)

                if currentProcessedYear > 0:
                    signal.saveKdbTableToDisk(q,year,KDB_ROOT)