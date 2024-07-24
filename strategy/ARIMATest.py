from strategy.ARIMATrader import ARIMATrader
from datetime import date
from time import time
from strategy.ARIMATrader import ARIMATraderParams, DEFAULT_ARIMA_PARAMS,acf,pacf,ARIMA,np,get_trading_days,_kdbdt,Dict
from strategy.base import Trader
import functools
import logging
import pandas as pd

logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class ARIMATestTrader(Trader):
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
        #returns[predCol] = np.nan
        #returns['predicted_se'] = np.nan
        #returns['model_order'] = np.nan

        test_trading_days = get_trading_days('NYSE', startDate, endDate)
        logging.info(f'Simulating for {len(test_trading_days)} trading days for {len(syms)} symbols: {",".join(syms)}')

        SKIP_FORECAST = True
        ACCESS = 'at'

        for index,td in enumerate(test_trading_days):
            if index % 5 == 0:
                logging.info(f'Simulating {index}/{len(test_trading_days)} for {td}')
            #panda_ts = _kdbdt(td)
            panda_ts = td
            SKIP_FORECAST = False
            JOIN_LATER = True
            if JOIN_LATER:
                addl = { x:[] for x in 'sym date prediction predicted_se model_order modelP modelQ'.split() }
            logging.warning('SKIPPING FORECAST')
            for sym in syms:
                if SKIP_FORECAST:
                    prediction,se = 0,0
                else:
                    forecast = models[sym].get_forecast(1)
                    prediction = forecast.predicted_mean[0]
                    se = forecast.se_mean[0] #standard error
                if JOIN_LATER:
                    mask = (returns.sym == sym) & (returns.date == panda_ts)
                    rowIndex = returns[mask].index
                    assert len(rowIndex)== 1, f'Expected 1 row got {len(rowIndex)}'
                    rowNum = rowIndex[0] # should be only one row
                    actual = returns.at[rowNum, self.params_.target]
                    addl['sym'].append(sym)
                    addl['date'].append(panda_ts)
                    addl[predCol].append(prediction)
                    addl['predicted_se'].append(se)
                    addl['model_order'].append(str(models[sym].model.order))
                    addl['modelP'].append(models[sym].model.order[0])
                    addl['modelQ'].append(models[sym].model.order[-1])
                else:
                    mask = (returns.sym == sym) & (returns.date == panda_ts)
                    rowIndex = returns[mask].index
                    assert len(rowIndex)== 1, f'Expected 1 row got {len(rowIndex)}'
                    rowNum = rowIndex[0] # should be only one row
                    actual = returns.at[rowNum, self.params_.target]
                    returns.at[rowNum, predCol] = prediction
                    returns.at[rowNum, 'predicted_se'] = se
                    returns.at[rowNum, 'model_order'] = str(models[sym].model.order)
                    returns.at[rowNum, 'modelP'] = models[sym].model.order[0]
                    returns.at[rowNum, 'modelQ'] = models[sym].model.order[-1]

                if not SKIP_FORECAST:
                    models[sym] = models[sym].append([actual])
        if JOIN_LATER:
            returns = returns.merge(pd.DataFrame(addl),how='left',on=['sym','date'])
        return returns

def run_test():
    TRAIN_START_DATE=date(2020,1,1)
    TRAIN_END_DATE=date(2020,6,30)
    TEST_START_DATE=date(2020,7,1)
    TEST_END_DATE=date(2020,7,31)

    trader = ARIMATestTrader()
    s = time()
    models = trader.calibrateModels(TRAIN_START_DATE,TRAIN_END_DATE)
    subset = list(models.keys())[0:300]
    subModels = {k:models[k] for k in subset}

    print(f'Calibration took {time()-s} seconds')
    s=time()
    trader.evaluateModels(subModels,TEST_START_DATE,TEST_END_DATE)
    print(f'Evaluation took {time()-s} seconds')

import cProfile
if __name__ == '__main__':
    cProfile.run('run_test()',"c:/temp/profile.txt")