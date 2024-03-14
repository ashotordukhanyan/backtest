from dataclasses import dataclass
import pandas as pd
from utils.utils import get_trading_days
import logging
from statsmodels.tsa.stattools import acf, pacf
from datetime import  date
import numpy as np
import functools
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict,List
from dateutil.relativedelta import relativedelta
from multiprocessing import freeze_support
from market_data.market_data import _kdbdt,get_md_conn
from utils.datagrid import CT, ColumnDef
import warnings
from utils.utils import cached_df
from strategy.base import Trader,TraderParams,TradingSignal
@dataclass
class ARIMATraderParams(TraderParams):
    alpha : float = 0.05 # significance level for ACF/PACF
    nlags : int = 5 # number of lags to consider for ACF/PACF

DEFAULT_ARIMA_PARAMS = ARIMATraderParams(target = 'c2cbn', universe = 'IWV')

class ARIMASignal(TradingSignal):
    _SCHEMA = [
        ColumnDef('sym',CT.SYMBOL, isKey=True),
        ColumnDef('date', CT.DATE, isKey=True),
        ColumnDef('target', CT.SYMBOL, isKey=True),
        ColumnDef('prediction', CT.F32), #prediction is assumed to be made for the "date" - i.e. after all of previous days data is available
        ColumnDef('predicted_se', CT.F32),
        ColumnDef('model_order', CT.STR),
        ColumnDef('modelP', CT.F32),
        ColumnDef('modelQ', CT.F32),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(date.today())]*len(frame)),
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.date.dt.year, isPartition=True),
    ]
    def __init__(self):
        super().__init__('arima_signal2', self._SCHEMA)
class ARIMATrader(Trader):
    ''' Auto-correlation based strategy '''
    def __init__(self,params : ARIMATraderParams = DEFAULT_ARIMA_PARAMS):
        self.params_ = params

    def _get_acf_result_unpacker(self, acf_or_pacf='acf', qstat: bool = True):
        ''' Returns lambda function that extracts data from result of acf function '''
        def unpack_acf_result(row):
            acf_result = row[f'{acf_or_pacf}_r']
            r = {}
            corrs = acf_result[0]
            for i in range(1, self.params_.nlags + 1):
                r[f'{acf_or_pacf}_{i}'] = corrs[i]
            if self.params_.alpha is not None:
                conf_intervals = acf_result[1]
                for i in range(1, self.params_.nlags + 1):
                    r[f'{acf_or_pacf}_{i}_ci_lower'] = conf_intervals[i][0] - corrs[i]
                    r[f'{acf_or_pacf}_{i}_ci_upper'] = conf_intervals[i][1] - corrs[i]
            if qstat:
                qstats = acf_result[2]
                pvalues = acf_result[3]
                for i in range(1, self.params_.nlags + 1):
                    r[f'{acf_or_pacf}_{i}_qstat'] = qstats[i - 1]
                    r[f'{acf_or_pacf}_{i}_pvalue'] = pvalues[i - 1]
            return pd.Series(r)

        return unpack_acf_result
    def getACFStats(self,returns:pd.DataFrame):
        # Calculate auto-correlations - trying to figure out how many of the timesteps have significant auto-correlations
        my_acf = functools.partial(acf, alpha=self.params_.alpha, nlags=self.params_.nlags, qstat=True)
        acf_results = returns.groupby('sym')[self.params_.target].apply(my_acf)
        acf_results.name = 'acf_r'
        acf_results = acf_results.to_frame()
        acf_stats = acf_results.apply(self._get_acf_result_unpacker('acf',True), axis=1, result_type='expand')

        my_pacf = functools.partial(pacf, alpha=self.params_.alpha, nlags=self.params_.nlags)
        pacf_results = returns.groupby('sym')[self.params_.target].apply(my_pacf)
        pacf_results.name = 'pacf_r'
        pacf_results = pacf_results.to_frame()
        pacf_stats = pacf_results.apply(self._get_acf_result_unpacker('pacf',False), axis=1, result_type='expand')
        acf_stats = acf_stats.join(pacf_stats)

        for l in range(1, self.params_.nlags):
            acf_stats[f'acf_{l}_inconf'] = ~(
                acf_stats[f'acf_{l}'].between(acf_stats[f'acf_{l}_ci_lower'], acf_stats[f'acf_{l}_ci_upper']))

        for l in range(1, self.params_.nlags):
            acf_stats[f'pacf_{l}_inconf'] = ~(
                acf_stats[f'pacf_{l}'].between(acf_stats[f'pacf_{l}_ci_lower'], acf_stats[f'pacf_{l}_ci_upper']))

        def num_leading_significant_lags(acf_or_pacf='acf'):
            def rowfunc(row):
                for l in range(1, self.params_.nlags):
                    if not row[f'{acf_or_pacf}_{l}_inconf']:
                        return l - 1
                return self.params_.nlags

            return rowfunc

        acf_stats['OPTP'] = acf_stats.apply(num_leading_significant_lags('acf'), axis=1)
        acf_stats['OPTQ'] = acf_stats.apply(num_leading_significant_lags('pacf'), axis=1)
        return acf_stats

    def calibrateModels(self,startDate:date,endDate:date):
        ''' Calibrate ARIMA models for all stocks for given period '''
        returns = self.getCleanData(startDate,endDate)
        acfStats = self.getACFStats(returns)
        significant = acfStats[(acfStats.OPTP > 0) | ((acfStats.OPTQ > 0) & (acfStats.acf_1_pvalue <= .05) & (
                acfStats.acf_2_pvalue <= .05) & ((acfStats.acf_3_pvalue <= .05)))]
        models = {}
        logging.info(f'Fitting ARIMA for {len(significant)} symbols: ')
        returns.set_index('date',inplace=True)
        for sym, data in significant.iterrows():
            sym_returns = returns.loc[returns.sym==sym,self.params_.target]
            try:
                mod = ARIMA(sym_returns.to_list(), order=(data.OPTP, 0, data.OPTQ))
                mod = mod.fit(method_kwargs={'warn_convergence': False})
                if mod.mle_retvals['converged']:  # warning gets printed - omit these for now
                    # logging.info(mod.summary())
                    models[sym] = mod
            except Exception as e:
                logging.error(f'Fit failed: for {sym} : {e}')
        logging.info(f'Done fitting ARMA')
        return models

    def evaluateModels(self, models: Dict[str, ARIMA], startDate:date, endDate:date) -> pd.DataFrame:
        syms = sorted(list(models.keys()))
        returns = self.getCleanData(startDate,endDate,symbols=syms,removeOutliers=False)
        syms = sorted(list(returns.sym.unique())) ## some symbols may have been excluded because of missing data
        predCol = 'prediction'
        #returns['actual'] = returns[self.params_.target]
        returns[predCol] = np.nan
        returns['predicted_se'] = np.nan
        returns['model_order'] = np.nan

        test_trading_days = get_trading_days('NYSE', startDate, endDate)
        logging.info(f'Simulating for {len(test_trading_days)} trading days for {len(syms)} symbols: {",".join(syms)}')

        for index,td in enumerate(test_trading_days):
            if index % 5 == 0:
                logging.info(f'Simulating {index}/{len(test_trading_days)} for {td}')
            panda_ts = _kdbdt(td)
            for sym in syms:
                forecast = models[sym].get_forecast(1)
                prediction = forecast.predicted_mean[0]
                se = forecast.se_mean[0] #standard error
                returns.loc[(returns.sym == sym) & (returns.date == panda_ts), predCol] = prediction
                returns.loc[(returns.sym == sym) & (returns.date == panda_ts), 'predicted_se'] = se
                returns.loc[(returns.sym == sym) & (returns.date == panda_ts), 'model_order'] = str(models[sym].model.order)
                returns.loc[(returns.sym == sym) & (returns.date == panda_ts), 'modelP'] = models[sym].model.order[0]
                returns.loc[(returns.sym == sym) & (returns.date == panda_ts), 'modelQ'] = models[sym].model.order[-1]
                actual = returns.loc[(returns.sym == sym) & (returns.date == panda_ts), self.params_.target].to_list()
                if len(actual) != 1:
                    raise Exception(f'{startDate} Expected 1 row got {len(actual)} for {sym} {td}')
                models[sym] = models[sym].append(actual)

        return returns

@cached_df
def simulate(TRAIN_PERIOD_START,TRAIN_PERIOD_END,TEST_PERIOD_START,TEST_PERIOD_END, codever=0) -> pd.DataFrame:
    logging.info(
        f'{index}/{len(SIM_DATES)} TRAINING {TRAIN_PERIOD_START} - {TRAIN_PERIOD_END} TESTING {TEST_PERIOD_START} - {TEST_PERIOD_END}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = trader.calibrateModels(TRAIN_PERIOD_START, TRAIN_PERIOD_END)
    simResults = trader.evaluateModels(models, TEST_PERIOD_START, TEST_PERIOD_END)
    simResults['asof_date'] = np.datetime64(date.today())
    simResults['year'] = simResults.date.apply(lambda x: x.year)
    simResults['target'] = trader.params_.target
    simResults = simResults[
        ['sym', 'date', 'prediction', 'target', 'predicted_se', 'model_order', 'modelP', 'modelQ', 'asof_date', 'year']]
    return simResults

if __name__ == '__main__':
    logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    DRY_RUN = True
    freeze_support()
    params = DEFAULT_ARIMA_PARAMS
    SIM_DATES = []
    SIM_DATE = date(2007,1,1)
    while ( SIM_DATE < date(2023,12,1) ):
        SIM_DATES.append(SIM_DATE)
        SIM_DATE  = SIM_DATE + params.trainFrequency

    trader = ARIMATrader(params)
    allSimResults = []
    currentProcessedYear = -1
    KDB_ROOT = 'C://KDB_MARKET_DATA2/'
    signal = ARIMASignal()
    if not DRY_RUN:
        with get_md_conn() as q:
            signal.kdbInitConnection(q)
    for index,SIM_DATE in enumerate(SIM_DATES):
        TRAIN_PERIOD_START = SIM_DATE - params.trainPeriod
        TRAIN_PERIOD_END = SIM_DATE - relativedelta(days=1)
        TEST_PERIOD_START = SIM_DATE
        TEST_PERIOD_END = SIM_DATE + params.trainFrequency# - relativedelta(days=1)

        simResults = simulate(TRAIN_PERIOD_START, TRAIN_PERIOD_END, TEST_PERIOD_START, TEST_PERIOD_END,2)
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