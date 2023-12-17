from eodhd import APIClient
from eodhd.apiclient import Interval
from datetime import date
from symbology.compositions import get_composition_history, get_broad_etfs, get_constituent_history
from utils.qpthack import qconnection,MetaData, qtemporal
from utils.utils import get_trading_days
import logging
import pandas as pd
from typing import List,Optional,Tuple
logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')
import re
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests_cache
from utils.utils import  get_trading_days

EODHD_API = '650b4b8c563d37.19655498'
def tpatchmissing(num_threads:int = 1,start_date=date(2000, 1, 1),end_date=date(2023,9,26)):
    requests_cache.install_cache()
    apic = APIClient(EODHD_API)
    with qconnection.QConnection('localhost', 12345, pandas=True) as q:
        p = q.sendSync(
            f'''select asc distinct date from eodhd_price where year >= {start_date.year},
            date >= {start_date.strftime("%Y.%m.%d")}, date <= {end_date.strftime("%Y.%m.%d")}''')
    trading_days = sorted(get_trading_days('NYSE',date(2000,1,1),date(2023,9,1)))
    vd = set(p.date[0].dt.date) #valid days
    missing_days = [ d for d in trading_days if d not in vd]
    download_dates(num_threads,apic,missing_days)
#    dates = get_trading_days('NYSE', start_date, end_date)
#    return download_dates(num_threads,apic,dates)

def tmain(num_threads:int = 1,start_date=date(2000, 1, 1),end_date=date(2023,9,26)):
    requests_cache.install_cache()
    apic = APIClient(EODHD_API)
    dates = get_trading_days('NYSE', start_date, end_date)
    return download_dates(num_threads,apic,dates)

def download_dates(num_threads:int,apic,dates):
    byYear = {}
    for d in dates:
        byYear[d.year] = byYear.setdefault(d.year,[]) + [d]

    years = list(reversed(sorted(byYear.keys())))
    #dates = list(reversed(sorted(dates))) ## lets go from the back

    #batches = [dates[offs: offs + num_threads] for offs in range(0, len(dates), num_threads)]

    logging.info(f'{len(dates)} to retrieve across {len(years)} years')

    for year in years:
        initQ(year)
        batch = list(reversed(sorted(byYear[year])))
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                executor.map(functools.partial(download_data,apic),batch)
        else:
            for trd_date in batch:
                download_data(apic,trd_date)

        print(f'Saving to disk')
        persistToDisk(year)


@functools.lru_cache(1)
def getUniverse(UNDERLYING_ETFS="IWB IWM IWV IVV".split(),START_DATE = date(2000,1,1),END_DATE = date(2023,9,1)) ->List[str]:
    BATCH_SIZE = 500
    ASOF_DATE = date.today()
    universe = set()
    for e in UNDERLYING_ETFS:
        comp_history = get_composition_history(e,START_DATE,END_DATE)
        universe = universe.union(set(comp_history.ticker.unique().tolist()))
    universe = sorted(list(universe.union(set(get_broad_etfs()))))
    universe = [u for u in universe if not re.findall(r'\d+|-', u)] #crap with digirs and - in ticker names
    logging.info(f"Got {len(universe)} symbols")
    return universe

def _df(data:dict)->Optional[pd.DataFrame]:
    return pd.DataFrame(data) if data is not None and len(data) else None

def download_data(apic:APIClient,trd_date:date):
    p,d,s = get_eod_price_split_dividends(apic,trd_date)
    streamToKDB(p,d,s,trd_date)

def get_eod_price_split_dividends(apic:APIClient,d:date) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    logging.warning(f'GETTING DATA FOR {d}')
    price_data = _df(apic.get_eod_splits_dividends_data(date = d))
    div_data = _df(apic.get_eod_splits_dividends_data(date = d, type='dividends'))
    splits_data = _df(apic.get_eod_splits_dividends_data(date = d, type='splits'))
    logging.warning(f'DONE GETTING DATA FOR {d}')
    return (price_data,div_data,splits_data)

_PRICE_META = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE, sym=qtemporal.QSYMBOL,volume=qtemporal.QLONG)
for f in 'open high low close adjusted_close'.split():
    _PRICE_META[f]=qtemporal.QFLOAT

_DIVS_META = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE, sym=qtemporal.QSYMBOL, currency=qtemporal.QSYMBOL,
                      dividend=qtemporal.QFLOAT, period=qtemporal.QSYMBOL,unadjustedValue=qtemporal.QFLOAT,
                      declarationDate=qtemporal.QDATE,recordDate=qtemporal.QDATE,paymentDate=qtemporal.QDATE)

_SPLITS_META = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE, sym=qtemporal.QSYMBOL) ##qtemporal.QSRING DOESNT WORK

def initQ(year: int):
    QCODE = f'''
if [ not `eodhd_price in key `.;
    eodhd_price :([ sym:`symbol$();date:`date$();year:`long$()] open:`real$();high:`real$();low:`real$();close:`real$();
            volume:`long$();adjusted_close:`real$();asof_date:`date$());
];

if [ not `eodhd_divs in key `.;
    eodhd_divs :([ sym:`symbol$();date:`date$();year:`long$()] dividend:`real$();currency:`symbol$(); 
                declarationDate:`date$();recordDate:`date$();paymentDate:`date$();period:`symbol$();unadjustedValue :`real$();asof_date:`date$() )
];


if [ not `eodhd_splits in key `.;
    eodhd_splits :([ sym:`symbol$();date:`date$();year:`long$()] split:();asof_date:`date$() )
];


eodhd_price_upd :`sym`date xkey (select from eodhd_price where year = {year:d}); 
eodhd_divs_upd : `sym`date xkey(select from eodhd_divs where year = {year:d}); 
eodhd_splits_upd :`sym`date xkey(select from eodhd_splits where year = {year:d});
'''
    #with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
    #   q.sendSync(QCODE)
    _sendSync(QCODE)

def persistToDisk(year:int):

    QCODE = '''
        MDROOT:"c:/KDB_MARKET_DATA2/";    
    '''
    for table in 'eodhd_price eodhd_divs eodhd_splits'.split():
        QCODE = QCODE + \
                f'''
            TABLEROOT: hsym `$(MDROOT,"{year:d}/","{table}/");
            TABLEROOT set .Q.en[hsym `$MDROOT] 0!{table}_upd;
        '''
    with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
        q.sendSync(QCODE)

def _dfs(df:pd.DataFrame):
    return 0 if df is None else df.shape[0]

def streamToKDB(price_data,div_data,splits_data, trd_date:date, asof_date = date.today(),MDROOT="c:/KDB_MARKET_DATA2/"):
    logging.warning(f'Streaming data for {trd_date} {_dfs(price_data)} price {_dfs(div_data)} divs \
        {_dfs(splits_data)} splits ')
    price_data.meta = _PRICE_META
    if div_data is not None:
        div_data.meta = _DIVS_META
    if splits_data is not None:
        splits_data.meta = _SPLITS_META

    for df in [price_data,div_data,splits_data]:
        if df is not None and not df.empty:
            df['asof_date']=pd.to_datetime(asof_date)
            df.rename(columns={'code':'sym'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            ##sanitize - remove columns with no meta
            extra_cols = [ c for c in df.columns if c not in df.meta.as_dict().keys() and c != "split" ] #bug workdoung
            if extra_cols:
                logging.debug(f'Dropping {extra_cols}')
            df.drop(columns=extra_cols, inplace=True)

    if div_data is not None:
        for dateField in 'declarationDate recordDate paymentDate'.split():
            div_data[dateField] = pd.to_datetime(div_data[dateField])

    if price_data is not None:
        _sendSync(f'{{[d] MDROOT:hsym `$"{MDROOT}";`eodhd_price_upd upsert .Q.en[MDROOT; 0!d];}}',price_data)
    if div_data is not None:
        _sendSync(f'{{[d] MDROOT:hsym `$"{MDROOT}";`eodhd_divs_upd upsert .Q.en[MDROOT; 0!d];}}',div_data)
    if splits_data is not None:
        _sendSync(f'{{[d] MDROOT:hsym `$"{MDROOT}";`eodhd_splits_upd upsert .Q.en[MDROOT; 0!d];}}',splits_data)

def _sendSync(qcode,*parameters):
    logging.warning(f'EXECUTING {qcode}')
    with qconnection.QConnection(host='localhost',port=12345,pandas=True) as q:
        q.sendSync(qcode,*parameters)

if __name__ == '__main__':
    #asyncio.run(amain(),debug=True)
    #tmain(start_date=date(2000,1,1),end_date=date(2001,12,31), num_threads=25)
    tpatchmissing(start_date=date(2000, 1, 1), end_date=date(2023, 9, 1), num_threads=25)
