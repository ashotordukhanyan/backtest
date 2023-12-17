import functools
from datetime import date
import pandas as pd
import requests
import json
from typing import List,Dict,Optional
from utils.utils import get_trading_month_ends, cached_df
import logging


## See URL for full list 'https://www.ishares.com/us/products/etf-investments#!type=ishares&style=All&view=keyFacts'


COMP_URLS={
'IWB' : 'https://www.ishares.com/us/products/239707/ishares-russell-1000-etf',
'IWM' : 'https://www.ishares.com/us/products/239710/ishares-russell-2000-etf',
'IWV' : 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf',
'IVV' : 'https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf'
}

SPX_FILE='c:/users/orduk/PycharmProjects/backtest/data/sp500historical.csv'

BROAD_ETFS = "SPY QQQ XLSR XLC XLY XLP XLE XLF XLV XLI XLB XLRE XLK XLU XAR KBE XBI KCE XHE XHS""\
 ""XHB KIE XWEB XME XES XOP XPH KRE XRT XSD XSW XTL XTN".split()

@functools.lru_cache
def get_spx_constituents_history()->pd.DataFrame:
    c=pd.read_csv(SPX_FILE)
    c['date']=pd.to_datetime(c['date']).dt.date
    return c

def get_spx_constituents(asof:date)->List[str]:
    c = get_spx_constituents_history()
    return c[c['date']<=asof].sort_values('date',ascending=False).head(1).iloc[0]['tickers'].split(',')


# -------------------------------------------------------------------------------------------------
def map_raw_item(unmapped_item):#
    return {
        'ticker': unmapped_item[0],
        'name': unmapped_item[1],
        'sector': unmapped_item[2],
        'asset_class': unmapped_item[3],
        'market_value': unmapped_item[4]['raw'],
        'weight': unmapped_item[5]['raw'],
        'notional_value': unmapped_item[6]['raw'],
        'shares': unmapped_item[7]['raw'],
        'cusip': unmapped_item[8],
        'isin': unmapped_item[9],
        'sedol': unmapped_item[10],
        'price': unmapped_item[11]['raw'],
        'location': unmapped_item[12],
        'exchange': unmapped_item[13],
        'currency': unmapped_item[14],
        'fx_rate': unmapped_item[15],
        'maturity': unmapped_item[16]
    }

def format_response(response_json) -> List[Dict]:
    input_items = response_json['aaData']
    output_items = []
    for input_item in input_items:
        mapped_item = map_raw_item(input_item)
        output_items.append(mapped_item)
    return (output_items)

def get_composition_history(etf_ticker:str, start_date:date, end_date:date, calendar: str = 'NYSE', equity_only=True) -> pd.DataFrame:
    '''Get composition history for a date range'''
    month_ends = get_trading_month_ends(calendar,start_date,end_date)
    data = []
    for m in month_ends:
        datum = get_composition(etf_ticker,m)
        data.append(datum)
    result = pd.concat(data)
    if equity_only:
        result = result[result.asset_class == 'Equity']
    return result

def get_constituent_history(ticker:str, etf_tickers:List[str], start_date:date, end_date:date, calendar: str = 'NYSE') -> Optional[pd.DataFrame]:
    '''Get composition history for a date range'''
    month_ends = get_trading_month_ends(calendar,start_date,end_date)
    data = []
    for m in month_ends:
        for etf_ticker in etf_tickers:
            datum = get_composition(etf_ticker,m)
            if datum is not None and not datum.empty:
                datum = datum[datum.ticker==ticker]
            if datum is not None and not datum.empty:
                data.append(datum)
    return  pd.concat(data) if data else None

def get_broad_etfs() -> List[str]:
    return BROAD_ETFS

@cached_df
def get_composition(etf_ticker:str, asof:date)-> pd.DataFrame:
    '''Get composition for a given ETF from ishares as of(must be a month end)'''
    #HACK - IWB comp are missing for 6 months and IWV missing from 2016-08-01 to 2017-01-01
    if etf_ticker == 'IWB' and asof.year == 2017 and asof<date(2017,7,31):
        asof = date(2016,12,30)
    if etf_ticker == 'IWV' and asof >= date(2017,1,31) and asof<=date(2017,6,30):
        asof = date(2016,12,30)
    if etf_ticker not in COMP_URLS:
        raise Exception('Unrecognized ticker '+etf_ticker)
    request_url = f'{COMP_URLS[etf_ticker]}/1467271812596.ajax?' \
                      f'fileType=json&tab=all&asOfDate={asof.strftime("%Y%m%d")}'
    logging.debug(f'requesting: {request_url}')
    response = requests.get(request_url)
    if response is None:
        raise Exception('Request timed out')
    if response.status_code != 200:
        raise Exception(f'Response status code {response.status_code}')
    response_json = json.loads(response.content)
    df = pd.DataFrame(format_response(response_json))
    df['asof_date'] = asof
    df['ETF'] = etf_ticker
    return df

#if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',filename=None)
    #a=get_composition_history('IWV',date(2020,1,31),date(2022,2,28))