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

def getCleanDataForUniverse(TRAIN_PERIOD_START,TRAIN_PERIOD_END,universe='IWB',benchmark='SPY',remove_outliers=True):
    comp = get_composition(universe,get_last_trading_month_end(TRAIN_PERIOD_END))
    tickers = sorted(comp.ticker.unique())
    return getCleanData(TRAIN_PERIOD_START,TRAIN_PERIOD_END,tickers,benchmark=benchmark,remove_outliers=remove_outliers)

def getCleanData(TRAIN_PERIOD_START,TRAIN_PERIOD_END,tickers,benchmark='SPY',remove_outliers=True):
    if benchmark is not None and benchmark not in tickers:
        tickers.append(benchmark)

    closes = get_adj_closes(tickers,TRAIN_PERIOD_START,TRAIN_PERIOD_END)

    logging.info(f'Got {closes.shape[0]} rows of data for {len(tickers)} requested tickers - {len(closes.sym.unique())} received tickers')
    trading_days = get_trading_days('NYSE',TRAIN_PERIOD_START,TRAIN_PERIOD_END)
    #remove from closes symbols that have less than trading_days rows
    original_syms = set(closes.sym.unique())
    closes = closes.groupby('sym').filter(lambda x: x.shape[0] == len(trading_days))
    #log symbols that were excluded
    excluded_syms = original_syms - set(closes.sym.unique())
    if benchmark in excluded_syms:
        logging.fatal(f'Excluded benchmark {benchmark}')
        raise Exception(f'Excluded benchmark {benchmark}')
    logging.info(f'Excluded {len(excluded_syms)} symbols with missing data: {",".join(sorted(excluded_syms))} ')
    logging.info(f'Excluded {len(excluded_syms)} symbols with missing data: {",".join(sorted(excluded_syms))} ')

    #calculate log returns without grouping - sort by sym,date then calc log ret of consecutive rows and finally correct to NA for rows
    # where sym changes
    closes.sort_values(['sym','date'],inplace=True)
    closes['prev_c2c'] = np.nan
    closes['c2c'] = np.nan
    closes['prev_volume'] = np.nan
    closes.loc[closes.sym == closes.sym.shift(1),'prev_volume'] = closes.volume.shift(1)
    closes.loc[closes.sym == closes.sym.shift(1),'c2c'] = (closes.adjusted_close - closes.adjusted_close.shift(1))/ closes.adjusted_close.shift(1)
    closes.loc[closes.sym == closes.sym.shift(2),'prev_c2c'] = (closes.adjusted_close.shift(1) - closes.adjusted_close.shift(2)) / closes.adjusted_close.shift(2)

    #closes_clean['log_return'] = np.log(closes_clean['adjusted_close']) - np.log(closes_clean['adjusted_close'].shift(1))
    #closes['c2c'] = (closes['adjusted_close']-closes['adjusted_close'].shift(1))/closes['adjusted_close'].shift(1)
    #closes.loc[closes.sym!=closes.sym.shift(1),'c2c']=np.nan

    closes.dropna(inplace=True)

    if remove_outliers:
        KNOWN_ETFS = set('SPY RTH XLF XLY XLP SMH XLI XLU XLV KRE IYR OIH XLK IYT XLE QQQ'.split())

        #remove "penny stocks" - stocks whose price dips below 1$ ever
        closes = closes.groupby('sym').filter(lambda x: x.adjusted_close.min()>1.)

        #remove outliers
        sym_stats=closes.groupby(['sym'], as_index=True)['c2c'].agg(['mean','std'])
        NUM_SD=3
        isEtf = sym_stats.index.isin(KNOWN_ETFS)
        meanTooBig = sym_stats["mean"].abs()>NUM_SD*sym_stats["mean"].std()
        meanTooWild = sym_stats["std"] > 2* NUM_SD * sym_stats["std"].std()
        outliers = sym_stats[ ~isEtf & (meanTooBig | meanTooWild)]
        logging.info(f'Excluded {len(outliers)} symbols with outlier returns: {",".join(sorted(outliers.index))} ')
        closes = closes[~closes.sym.isin(outliers.index)]

    if benchmark is not None:
        if benchmark not in  closes.sym.unique():
            logging.fatal(f'Excluded benchmark {benchmark} because of data quality')
            raise Exception(f'Excluded benchmark {benchmark} because of data quality')

        #create new columns c2cdn and o2cdn which will contain c2c - c2c of benchmark and o2c- o2c of benchmark for the same date
        benchmark_closes = closes[closes.sym==benchmark][['date','sym','c2c','o2c','prev_c2c']]
        benchmark_closes.rename(columns={'sym':'benchmark','c2c':'c2cb','o2c':'o2cb','prev_c2c':'prev_c2cb'},inplace=True)
        closes['benchmark'] = benchmark
        closes = closes.merge(benchmark_closes, on=['date', 'benchmark'], how='left')
        closes['c2cdn'] = closes['c2c'] - closes['c2cb']
        closes['prev_c2cdn'] = closes['prev_c2c'] - closes['prev_c2cb']
        closes['o2cdn'] = closes['o2c'] - closes['o2cb']
        closes = closes[closes.sym!=benchmark] ##remove benchmark itself because it will always have c2cdn=0
    return closes

def get_acf_result_unpacker(nlags:int, alpha:bool,qstat:bool):
    def unpack_acf_result(row):
        acf_result = row['acf_r']
        r = {}
        corrs = acf_result[0]
        for i in range(1,nlags+1):
            r[f'acf_{i}'] = corrs[i]
        if alpha is not None:
            conf_intervals = acf_result[1]
            for i in range(1,nlags + 1):
                r[f'acf_{i}_ci_lower'] = conf_intervals[i][0] - corrs[i]
                r[f'acf_{i}_ci_upper'] = conf_intervals[i][1] - corrs[i]
        if qstat:
            qstats = acf_result[2]
            pvalues = acf_result[3]
            for i in range(1,nlags + 1):
                r[f'acf_{i}_qstat'] = qstats[i-1]
                r[f'acf_{i}_pvalue'] = pvalues[i-1]
        return pd.Series(r)
    return unpack_acf_result
def get_pacf_result_unpacker(nlags:int, alpha:bool):
    def unpack_pacf_result(row):
        pacf_result = row['pacf_r']
        r = {}
        corrs = pacf_result[0]
        for i in range(1,nlags+1):
            r[f'pacf_{i}'] = corrs[i]
        if alpha is not None:
            conf_intervals = pacf_result[1]
            for i in range(1,nlags + 1):
                r[f'pacf_{i}_ci_lower'] = conf_intervals[i][0] - corrs[i]
                r[f'pacf_{i}_ci_upper'] = conf_intervals[i][1] - corrs[i]
        return pd.Series(r)
    return unpack_pacf_result

def get_acf_stats(closes,ALPHA,QSTAT,NLAGS,target):
    #Calculate auto-correlations
    my_acf = functools.partial(acf,alpha=ALPHA,nlags=NLAGS,qstat=QSTAT)
    acf_results = closes.groupby('sym')[target].apply(my_acf)
    acf_results.name='acf_r'
    acf_results = acf_results.to_frame()
    acf_stats = acf_results.apply(get_acf_result_unpacker(NLAGS,ALPHA,QSTAT),axis=1,result_type='expand')

    my_pacf = functools.partial(pacf, alpha=ALPHA, nlags=NLAGS)
    pacf_results = closes.groupby('sym')[target].apply(my_pacf)
    pacf_results.name = 'pacf_r'
    pacf_results = pacf_results.to_frame()
    pacf_stats = pacf_results.apply(get_pacf_result_unpacker(NLAGS, ALPHA), axis=1, result_type='expand')
    acf_stats= acf_stats.join(pacf_stats)

    for l in range(1,NLAGS):
        acf_stats[f'acf_{l}_inconf'] = ~(acf_stats[f'acf_{l}'].between(acf_stats[f'acf_{l}_ci_lower'],acf_stats[f'acf_{l}_ci_upper']))

    for l in range(1,NLAGS):
        acf_stats[f'pacf_{l}_inconf'] = ~(acf_stats[f'pacf_{l}'].between(acf_stats[f'pacf_{l}_ci_lower'],acf_stats[f'pacf_{l}_ci_upper']))

    def num_leading_significant_lags(acf_or_pacf='acf'):
        def rowfunc(row):
            for l in range(1,NLAGS):
                if not row[f'{acf_or_pacf}_{l}_inconf']:
                    return l-1
            return NLAGS
        return rowfunc

    acf_stats['OPTP']=acf_stats.apply(num_leading_significant_lags('acf'),axis=1)
    acf_stats['OPTQ']=acf_stats.apply(num_leading_significant_lags('pacf'),axis=1)
    return acf_stats

def extract_target(closes,sym,target):
    cd = closes.set_index('date')
    return cd.loc[cd.sym==sym,target]

def csv_show(data):
    data.to_csv('c:/temp/temp.csv',mode="w+")
    os.system( 'start "C:\Program Files\Microsoft Office 15\root\office15\EXCEL.EXE" "c:/temp/temp.csv" ')

def calibrateModels(prices,acf_stats,target):
    significant = acf_stats[(acf_stats.OPTP > 0) | ( (acf_stats.OPTQ > 0) & (acf_stats.acf_1_pvalue <= .05) & (
                acf_stats.acf_2_pvalue <= .05) & ((acf_stats.acf_3_pvalue <= .05)) )]
    models={}
    logging.info(f'Fitting ARIMA for {len(significant)} symbols: ')
    for sym,data in significant.iterrows():
        sym_returns = extract_target(prices,sym,target)
        try:
            mod = ARIMA(sym_returns.to_list(), order=(data.OPTP,0,data.OPTQ))
            mod = mod.fit(method_kwargs={'warn_convergence': False})
            if mod.mle_retvals['converged']: #warning gets printed - omit these for now
                #logging.info(mod.summary())
                models[sym]=mod
        except Exception as e:
            logging.error(f'Fit failed: for {sym} : {e}')
    logging.info(f'Done fitting ARMA')
    return models


def simulate(models: Dict[str, ARIMA], TEST_PERIOD_START, TEST_PERIOD_END,benchmark,target='c2cdn') -> pd.DataFrame:
    logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s',
                        datefmt='%H:%M:%S')
    test_syms = sorted(list(models.keys()))

    test_closes = getCleanData(get_previous_trading_day(TEST_PERIOD_START,days_back=2), TEST_PERIOD_END, test_syms,benchmark,False)
     ##getting prev to get returns andd prev returns for first day

    test_syms = sorted(list(test_closes.sym.unique())) ## some symbols may have been excluded because of missing data
    predCol = f'predicted_{target}'
    test_closes[predCol] = np.nan
    test_closes['predicted_se'] = np.nan
    test_closes['model_order'] = np.nan

    test_trading_days = get_trading_days('NYSE', TEST_PERIOD_START, TEST_PERIOD_END)
    logging.info(f'Simulating for {len(test_trading_days)} trading days for {len(test_syms)} symbols: {",".join(test_syms)}')

    for td in test_trading_days:
        panda_ts = pd.to_datetime(td)
        for sym in test_syms:
            forecast = models[sym].get_forecast(1)
            prediction = forecast.predicted_mean[0]
            se = forecast.se_mean[0]
            test_closes.loc[
                (test_closes.sym == sym) & (test_closes.date == panda_ts), predCol] = prediction
            test_closes.loc[
                (test_closes.sym == sym) & (test_closes.date == panda_ts), 'predicted_se'] = se
            test_closes.loc[
                (test_closes.sym == sym) & (test_closes.date == panda_ts), 'model_order'] = str(models[sym].model.order)
            actual = test_closes.loc[(test_closes.sym == sym) & (test_closes.date == panda_ts), target].to_list()
            if len(actual) != 1:
                raise Exception(f'{TEST_PERIOD_START} Expected 1 row got {len(actual)} for {sym} {td}')
            models[sym] = models[sym].append(actual)
    return test_closes

from utils.utils import cached_df
@cached_df
def simulateTradeDate(TRADE_DATE:date, ALPHA, NLAGS,universe='IWB',benchmark='SPY',codever=0)->pd.DataFrame :
    logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s',datefmt='%H:%M:%S')
    logging.warning(f'TRADE DATE {TRADE_DATE.strftime("%Y-%m-%d")}' )
    TRAIN_PERIOD_START = TRADE_DATE
    TRAIN_PERIOD_END = TRAIN_PERIOD_START +  relativedelta(months=6) - relativedelta(days=1)
    TEST_PERIOD_START = TRAIN_PERIOD_END + timedelta(days=1)
    TEST_PERIOD_END= TEST_PERIOD_START+ relativedelta(months=1) - relativedelta(days=1)

    logging.debug('Getting data')
    closes = getCleanDataForUniverse(TRAIN_PERIOD_START,TRAIN_PERIOD_END,universe=universe,benchmark=benchmark)
    logging.debug('Done')
    #Calculate auto-correlations
    #ALPHA=.05
    QSTAT = True
    #NLAGS=3
    logging.debug('Calibratiing AC stats')
    target = 'c2cdn'
    acf_stats =  get_acf_stats(closes,ALPHA,QSTAT,NLAGS,target=target)
    logging.debug('Done')

    models = calibrateModels(closes,acf_stats,target=target)
    simResults = simulate(models,TEST_PERIOD_START,TEST_PERIOD_END,benchmark=benchmark,target=target)
    return simResults

def analyze():
    from sklearn.pipeline import make_pipeline,Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    ##sim = pd.read_csv('c:/temp/SIM_RESULTS_11_18.csv')
    sim = pd.read_csv('c:/temp/SIM_RESULTS_11_22.csv')
    features = ['predicted_log_return','predicted_se','model_order']
    X = sim[features]
    Y = sim['log_return']
    ct = make_column_transformer( (OneHotEncoder(),['model_order']),remainder="passthrough")
    rf = RandomForestRegressor(n_estimators=100,max_depth=4,verbose=True)
    pipe = make_pipeline(ct,rf)
    pipe.fit(X,Y)
    fn = ct.get_feature_names_out()
    fi = rf.feature_importances_
    fi_df = pd.DataFrame({'feature':fn,'importance':fi})
    sim['simple_return'] = sim.log_return * np.sign(sim.predicted_log_return)
    sim['P'] = sim.model_order.apply(lambda x: eval(x)[0])
    sim['Q'] = sim.model_order.apply(lambda x: eval(x)[-1])
    stats=sim.groupby('model_order').agg({'log_return': ['mean', 'std', 'min', 'max', 'count']})
    statsP=sim.groupby('P').agg({'log_return': ['mean', 'std', 'min', 'max', 'count']})
    statsQ=sim.groupby('Q').agg({'log_return': ['mean', 'std', 'min', 'max', 'count']})
    hm = pd.pivot_table(sim, values='log_return', index='P', columns='Q', aggfunc='mean')
if __name__ == '__main__':
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
    ALPHA = .05
    NLAGS = 5
    if PARRALEL>1:
        distFunc = functools.partial(simulateTradeDate,ALPHA=ALPHA,NLAGS=NLAGS,universe='IWV',codever=20240108)
        with Pool(PARRALEL) as p:
            allSimResults = p.map(distFunc, TRADE_DATES)
            #allSimResults = p.map(simulate, TRADE_DATES)
    else:
        for TRADE_DATE in TRADE_DATES:
            simResults = simulateTradeDate(TRADE_DATE, ALPHA, NLAGS,universe='IWV',codever=20240108+1)
            TRADE_DATE  = TRADE_DATE + TIME_STEP
            allSimResults.append(simResults)

    #allSimResults = pd.concat(allSimResults)
    #allSimResults.to_csv('c:/temp/SIM_RESULTS_01_08.csv',mode="w+")