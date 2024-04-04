import logging
from datetime import date
from typing import List
import pandas_market_calendars as mcal
import pandas as pd
import calendar
import functools
import datetime

def _string_key(arg) -> str :
    if isinstance(arg,str):
        return arg
    if isinstance(arg,datetime.date):
        return arg.strftime('%Y%m%d')
    if isinstance(arg,(int,float)):
        return str(arg)
    if isinstance(arg, (list,tuple)):
        return _KEY_DELIMETER.join([str(x) for x in arg])
    else:
        raise Exception(f'Unknown type {type(arg)}')

_KEY_DELIMETER = '_'
_CACHE_LOC = 'c:/temp/DFCACHE'
_RETRY_ON_EMPTY = True #TODO CHANGE
_DISABLED_DF_FUNCS_ = set()
def disable_df_cache(func):
    _DISABLED_DF_FUNCS_.add(func.__name__)
    return func

def enable_df_cache(func):
    _DISABLED_DF_FUNCS_.remove(func.__name__)
    return func

def cached_df(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """DF CACHING WRAP FUNCTION"""
        keys = [func.__name__]+ [_string_key(a) for a in args]
        for k,v in kwargs.items():
            if not k.startswith("_"):
                keys.append(f'{k}_{_string_key(v)}')
        key=_KEY_DELIMETER.join(keys)
        fname = f'{_CACHE_LOC}/{key}.gz'
        try:
            result = pd.read_pickle(fname)
            if _RETRY_ON_EMPTY and result.empty:
                raise Exception(f'Empty dataframe for key {key}')
            logging.debug(f'Cache hit for key {key}')
            return result
        except:
            logging.debug(f'Cache miss for key {key} file {fname}')
            result = func(*args,**kwargs)
            if result is not None:
                if not isinstance(result,pd.DataFrame):
                    raise Exception(f'Expected dataframe got {type(result)}')
                result.to_pickle(fname)
                return result
    if func.__name__ in _DISABLED_DF_FUNCS_:
        return func
    return wrapper

def end_of_month(d: date)-> date:
    '''End of calendar month for a given date'''
    return date(d.year,d.month,calendar.monthrange(d.year,d.month)[1])

def get_trading_month_ends(exchange_code:str, start_date:date, end_date:date)->List[date]:
    '''Enf of month trading days for a given exchange that fall between start and end date parameters'''
    if exchange_code not in mcal.get_calendar_names():
        raise Exception(f"Unrecognized echange code {exchange_code}")
    trading_days = mcal.get_calendar(exchange_code).valid_days(start_date,end_of_month(end_date))
    df = pd.DataFrame(index=trading_days,data=trading_days,columns=['trading_date'])
    eom_dates = df.groupby([df.index.year,df.index.month]).max()['trading_date'].dt.date
    return  [x for x in eom_dates if x>=start_date and x<= end_date]

@functools.lru_cache(1)
def get_all_trading_month_ends(exchange_code:str='NYSE')->dict :
    return {(d.year,d.month):d for d in get_trading_month_ends(exchange_code,date(1900,1,1),date(2030,1,1))}

def get_last_trading_month_end(d:date,exchange_code:str = 'NYSE')->date:
    '''Last trading month end for a given exchange for a given year/month'''
    all_me = get_all_trading_month_ends(exchange_code)
    if all_me[(d.year,d.month)] <= d:
        return all_me[(d.year,d.month)]
    else:
        prev = (d.year,d.month-1) if d.month>1 else (d.year-1,12)
        return all_me[prev]


    return get_trading_month_ends(exchange_code,date(year,month,1),date(year,month,calendar.monthrange(year,month)[1]))[0]

def get_trading_days(exchange_code:str, start_date:date, end_date:date)->List[date]:
    '''Trading days between start/end '''
    if exchange_code not in mcal.get_calendar_names():
        raise Exception(f"Unrecognized echange code {exchange_code}")
    trading_days = mcal.get_calendar(exchange_code).valid_days(start_date,end_date)
    return trading_days.date if trading_days is not None else None

def get_next_trading_day(adate:date,exchange_code:str='NYSE')->date:
    '''Next trading day for a given exchange after a given date'''
    if exchange_code not in mcal.get_calendar_names():
        raise Exception(f"Unrecognized echange code {exchange_code}")
    trading_days = mcal.get_calendar(exchange_code).valid_days(adate,adate+datetime.timedelta(days=7))
    return trading_days.date[0] if trading_days is not None else None
def get_next_trading_days(dates:List[date], exchange_code:str='NYSE')->date:
    '''Next trading days for many days for a given exchange '''
    if exchange_code not in mcal.get_calendar_names():
        raise Exception(f"Unrecognized echange code {exchange_code}")
    trading_days = mcal.get_calendar(exchange_code).valid_days(min(dates),max(dates)+datetime.timedelta(days=7)).date
    #for each date in dates get first trading_day that is >= date
    trading_day_index=0
    mapping = {} #date to next trading day mapping
    for d in sorted(dates):
        while(trading_days[trading_day_index] <= d):
            trading_day_index+=1
        mapping[d]=trading_days[trading_day_index]
    return [mapping[d] for d in dates]

@functools.lru_cache(50)
def get_previous_trading_day(adate:date,days_back:int=1, exchange_code:str='NYSE'):
    cal = mcal.get_calendar(exchange_code)
    date = adate - pd.tseries.offsets.CustomBusinessDay(days_back, holidays=cal.holidays().holidays)
    return date.to_pydatetime().date()


from time import perf_counter
from contextlib import ContextDecorator
class TimeMe(ContextDecorator):
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        self.time = perf_counter()
        return self
    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.time
        print(f'{self.msg} took {elapsed:.3f} seconds')