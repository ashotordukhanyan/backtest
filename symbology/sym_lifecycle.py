from symbology.compositions import get_composition_history
from utils.utils import cached_df
from typing import List
from datetime import date
from utils.utils import get_trading_days
from utils.qpthack import QConnection
from tqdm import tqdm
from datetime import timedelta
import pandas as pd

def _kdt(d:date)->str :
    return d.strftime('%Y.%m.%d')

def get_all_syms_active_on_date(asof:date)->List[str]:
    lc = get_sym_lifecycle()
    lc = lc[(lc.start_date<=asof) & (lc.end_date>=asof)]
    return sorted(lc.sym.unique())

def get_all_syms_active_on_period(period_start:date,period_end:date)->List[str]:
    lc = get_sym_lifecycle()
    lc = lc[(lc.start_date<=period_start) & (lc.end_date>=period_end)]
    return sorted(lc.sym.unique())

def is_sym_active_on_date(sym:str,asof:date)->bool:
    lc = get_sym_lifecycle()
    lc = lc[(lc.sym==sym) & (lc.start_date<=asof) & (lc.end_date>=asof)]
    return lc.shape[0]!=0

def is_sym_active_on_period(sym:str,period_start:date,period_end:date)->bool:
    lc = get_sym_lifecycle()
    lc = lc[(lc.sym==sym) & (lc.start_date<=period_start) & (lc.end_date>=period_end)]
    return lc.shape[0]!=0

@cached_df
def get_sym_lifecycle(ASD = date(2000,1,1),AED = date(2023,9,1)) -> pd.DataFrame:
    trading_days = sorted(get_trading_days('NYSE',ASD,AED))
    trading_days_ix = {td:i for i,td in enumerate(trading_days)}

    qc = f'select asc distinct date by sym from eodhd_price where year >= {ASD.year},date>={_kdt(ASD)},date<={_kdt(AED)}, adjusted_close>0, adjusted_close <>1000000.0, volume>0'

    with QConnection('localhost', 12345, pandas=True) as q:
        p = q.sendSync(qc)



    syms = sorted(p.index.to_list())
    res = {}
    for sym in tqdm(syms):
        svd = sorted(p.loc[sym]['date'].dt.date) #stock valid days
        periods = [ [svd[0]] ]
        for i in range(1,len(svd)):
            if (trading_days_ix[svd[i]] - trading_days_ix[svd[i-1]] == 1): # no gap
                periods[-1].append(svd[i])
            else:
                #gap !!!
                periods.append([svd[i]])
        MIN_PERIOD_DAYS = 30
        pse = [ (x[0],x[-1]) for x in periods if (x[-1]-x[0])>timedelta(days=MIN_PERIOD_DAYS)] #periods start end
        if pse:
            res[sym.decode()]=pse
    ##convert res to dataframe
    table=[]
    for sym,periods in res.items():
        for ix,p in enumerate(periods):
            table.append([sym,ix,p[0],p[1]])
    resdf = pd.DataFrame(table,columns=['sym','period_ix','start_date','end_date'])

    return resdf