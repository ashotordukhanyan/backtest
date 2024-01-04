import numpy as np

from symbology.compositions import get_composition_history
from utils.utils import cached_df
from typing import List,Optional
from datetime import date
import yfinance as yf
import logging
import pandas as pd
from functools import lru_cache
from utils.qpthack import QConnection

@cached_df
def _get_daily_market_data_stripe(stripe:str, start_date:date, end_date:date,_tickers: List[str]):
    logging.info(f"Retrieving market data for {len(_tickers)} symbols for stripe {stripe}")
    market_data = yf.download(list(_tickers),start=start_date.strftime('%Y-%m-%d'),end=end_date.strftime('%Y-%m-%d'),
                              actions=True,auto_adjust=True,progress=True)
    logging.info(f'Retrieved data of shape {market_data.shape}')
    market_data['asof_date'] = date.today()
    return market_data

def get_daily_market_data(universes:List[str], start_date:date, end_date:date):
    tickers = set()
    for u in universes:
        compositions = get_composition_history(u,start_date,end_date)
        tickers = tickers.union(set(compositions.ticker.unique()))

    striped = {}
    for t in tickers:
        striped.setdefault(t[0],[]).append(t)

    stripes = sorted(striped.keys())
    dfs = []
    for s in stripes:
        dfs.append( _get_daily_market_data_stripe(s,start_date,end_date,_tickers = striped[s]))
    concat = pd.concat(dfs,axis=1)
    return concat

@lru_cache(1)
def get_md_conn()->QConnection:
    return QConnection(host='localhost',port=12345,pandas=True)

def get_all_syms(start_date:Optional[date]=None,end_date:Optional[date]=None)->List[str]:
    start_date = start_date or date.min
    end_date = end_date or date.max

    with get_md_conn() as q:
        res = q.sendSync('{ [sd;ed] select distinct sym from eodhd_price where year>=`year$sd,year<=`year$ed, date>=sd, date<=ed}',_kdbdt(start_date),_kdbdt(end_date))
    return [x.decode() for x in res.sym]

def _kdbdt(d:date)-> np.datetime64:
    return np.datetime64(d)

def get_price_volume(syms:List[str],start_date:date,end_date:date)->pd.DataFrame:
    with get_md_conn() as q:
        res = q.sendSync(
            '''{[syms;sd;ed] 
                select date,sym,adjusted_close,`float$volume from eodhd_price where year>=`year$sd,year<=`year$ed, date>=sd,date<=ed,sym in `$syms}''',
                         syms,_kdbdt(start_date),_kdbdt(end_date))
    #cast volume to float - workaround qpython serilization bug
    res['sym'] = res.sym.str.decode('utf-8')
    return res

def get_adj_closes(syms:List[str],start_date:Optional[date]=None,end_date:Optional[date]=None,check_volume:bool = True )->pd.DataFrame:
    start_date = start_date or date.min
    end_date = end_date or date.max
    vcheck = '' if not check_volume else ',volume>0'
    query = f'{{[syms;sd;ed] \
                select date,sym,adjusted_close,o2c:(close-open)%open from eodhd_price where date>=sd,date<=ed,sym in `$syms {vcheck} }}'

    with get_md_conn() as q:
        res = q.sendSync(query, syms,_kdbdt(start_date),_kdbdt(end_date))
    #cast volume to float - workaround qpython serilization bug
    res['sym'] = res.sym.str.decode('utf-8')
    return res

def get_daily_stats(tbl:pd.DataFrame) -> pd.DataFrame:
    '''Calculate daily stats for a given dataframe of dates and syms'''
    with get_md_conn() as q:
        res = q.sendSync(
            "{[tbl] (select `date$date, `sym$sym from tbl) lj (`sym`date xkey select sym,date,ar_vol,ar_beta,ADV,ADVD from daily_stats) }",
            tbl[['date', 'sym']])
    res['sym'] = res.sym.str.decode('utf-8')
    return res


