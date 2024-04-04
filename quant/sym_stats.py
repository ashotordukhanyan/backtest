from market_data.market_data import get_all_syms,get_md_conn
import logging
import numpy as np
from utils.qpthack import qconnection,qtemporal,MetaData
import pandas as pd
from datetime import date
from tqdm import tqdm
from typing import List
from utils.utils import get_previous_trading_day
from market_data.market_data import _kdbdt
#@DeprecationWarning('Use symstats.py instead')

MDROOT="c:/KDB_MARKET_DATA2/"
@DeprecationWarning('Use symstats.py instead')
def get_md(syms:List[str],SD:date,ED:date)->pd.DataFrame:
    with get_md_conn() as q:
        md = q.sendSync(
            '{[syms;sd;ed] select date,sym,`float$volume,dvolume:close*volume, adjusted_close from eodhd_price where date>=sd,date<=ed,sym in `$syms,volume>0}',
                syms,_kdbdt(SD),_kdbdt(ED))
    # NB cast to float needed because returning int from kdb triggers some qpython pandas integration bug
    md['sym'] = md.sym.str.decode('utf-8')
    # md.set_index(['Ticker', 'date'], drop=True, inplace=True)
    md.sort_values(['sym', 'date'], inplace=True)
    md.loc[md['sym'] == md['sym'].shift(), 'a_return'] = \
        (md['adjusted_close'] - md['adjusted_close'].shift(1)) / md['adjusted_close'].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        md['log_return'] = np.where((md['a_return']<=-1.)|(md['a_return'].isna()),np.nan,np.log(1. + md['a_return']))
    return md

@DeprecationWarning('Use symstats.py instead')
def std_wo_outliers(x):
    return x[(x>x.quantile(0.05)) & (x<x.quantile(0.95))].std()
@DeprecationWarning('Use symstats.py instead')
def _sendSync(qcode,*parameters):
    logging.info(f'EXECUTING {qcode}')
    with qconnection.QConnection(host='localhost',port=12345,pandas=True) as q:
        q.sendSync(qcode,*parameters)

_STATS_META = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE, sym=qtemporal.QSYMBOL, year=qtemporal.QLONG)
for f in 'lr_vol ar_vol ADV ADVD ar_beta'.split():
    _STATS_META[f]=qtemporal.QFLOAT

@DeprecationWarning('Use symstats.py instead')
def initQ(year: int):
    QCODE = f'''
        $[ not `daily_statsREMOVEME in key `.;
            [ 
                daily_stats :([ sym:`symbol$();date:`date$();year:`long$()] lr_vol:`real$();ar_vol:`real$();ADV:`real$();ADVD:`real$();ar_beta:`real$();asof_date:`date$());
                daily_stats_upd :`sym`date xkey (select from daily_stats );
            ];
            [ daily_stats_upd :`sym`date xkey (select from daily_stats where year = {year:d}); ]
        ];
    '''
    #with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
    #   q.sendSync(QCODE)
    _sendSync(QCODE)

@DeprecationWarning('Use symstats.py instead')
def calc_and_store_yearly_stats(year:int):
    logging.info(f'{year} Initialising Q')
    initQ(year)
    logging.info(f'{year} Calculating stats')
    data = calc_yearly_stats(year)
    if data is not None and len(data) > 0:
        data.meta = _STATS_META
        extra_cols = [c for c in data.columns if c not in data.meta.as_dict().keys()]
        if extra_cols:
            logging.debug(f'Dropping {extra_cols}')
            data.drop(columns=extra_cols, inplace=True)
        _sendSync(f'{{[d] MDROOT:hsym `$"{MDROOT}";`daily_stats_upd upsert .Q.en[MDROOT; 0!d];}}',data)
        logging.info(f'{year} Persisting to disk')
        persistToDisk(year)

@DeprecationWarning('Use symstats.py instead')
def persistToDisk(year:int):
    QCODE = f' MDROOT:"{MDROOT}";'
    QCODE = QCODE + \
            f'''
        TABLEROOT: hsym `$(MDROOT,"{year:d}/","daily_stats/");
        TABLEROOT set .Q.en[hsym `$MDROOT] 0!daily_stats_upd;
    '''
    _sendSync(QCODE)

@DeprecationWarning('Use symstats.py instead')
def calc_yearly_stats(year:int):
    ##Get a list of all symbols from kdb
    universe = sorted(get_all_syms(date(year,1,1),date(year,12,31)))
    #TEMP REMOVE
    #universe = universe[0:10]
    logging.info(f'{year} Universe has {len(universe)} symbols')
    MD_SD = get_previous_trading_day(date(year,1,1),300) #252 should suffice but just to be safe
    MD_ED = date(year,12,31)
    BATCH_SIZE=500
    ASOF_DATE = date.today()
    logging.info(f'Getting benchmark market data symbols')
    benchmark_md = get_md(['SPY'],MD_SD,MD_ED).set_index('date')
    logging.info(f'Getting market data')
    md = get_md(universe,MD_SD,MD_ED)
    logging.info(f'Done getting market data')

    md['lr_vol'] = md.groupby('sym')['log_return'].apply(lambda x: x.rolling(252,min_periods=252).std()).reset_index(level=0,drop=True)
    md['ar_vol'] = md.groupby('sym')['a_return'].apply(
        lambda x: x.rolling(252, min_periods=252).std()).reset_index(level=0, drop=True)
    md['ADV'] = md.groupby('sym')['volume'].apply(
        lambda x: x.rolling(21, min_periods=21).median()).reset_index(level=0, drop=True)
    md['ADVD'] = md.groupby('sym')['dvolume'].apply(
        lambda x: x.rolling(21, min_periods=21).median()).reset_index(level=0, drop=True)
    md['asof_date'] = pd.to_datetime(ASOF_DATE)
    ##join md and benchmark to get benchmark adjusted returns
    md = md.merge(benchmark_md[['log_return','a_return']],how='left',left_on='date',right_index=True,suffixes=('','_benchmark'))
    md.sort_values(['sym', 'date'],inplace=True)
    md.reset_index(drop=True,inplace=True)

    ar_vcv = md.groupby('sym').rolling(252, min_periods=252)[['a_return', 'a_return_benchmark']].cov()['a_return_benchmark']
    ar_covariance = ar_vcv.loc[:, :, 'a_return'] #covariance of sym arithmetic returns with benchmark returns
    benchmark_ar_var = ar_vcv.loc[:, :, 'a_return_benchmark']
    ar_beta = (ar_covariance / benchmark_ar_var) # key is (sym, index) tuple where index is a pointer to original md frame
    md['ar_beta'] = ar_beta.droplevel(0)
    md['year'] = md.date.dt.year
    return md[md.year==year]

# if __name__ == '__main__':
#     logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s',datefmt='%H:%M:%S')
#     for year in tqdm(range(2020,2024)):
#         calc_and_store_yearly_stats(year)
