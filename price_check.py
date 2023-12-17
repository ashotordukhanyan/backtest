from datetime import date
import logging
from symbology.compositions import get_composition_history
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')
from utils.utils import get_trading_days
from utils.qpthack import qconnection

def parr(a,max_len=10):
    if a is not None and len(a)> max_len:
        return(f'[...{len(a)} elements...]')
    else:
        return str(a)

r1 = get_composition_history('IVV',date(2000,1,1),date.today())

for comp_date in [sorted(r1.asof_date.unique())[0]]:
    logging.info(f'Working on asof {comp_date}')
    composition = r1[r1.asof_date==comp_date]
    syms = sorted(composition.ticker.unique())
    training_period = ( comp_date + relativedelta(months=-6),comp_date + relativedelta(days=-1) )
    test_period = ( comp_date + relativedelta(days=1),comp_date + relativedelta(months=3) )
    import numpy as np
    trading_days = get_trading_days('NYSE',training_period[0],test_period[1])
    qtrading_days = [np.datetime64(x) for x in trading_days]
    #optimize kdb data access using year partition var
    years = list(set([x.year for x in trading_days]))
    with qconnection.QConnection('localhost',12345,pandas=True) as q:
        data = q.sendSync('{[y;d;s] select date,sym,adjusted_close from eodhd_price where year in y,sym in `$s}',years,
                          qtrading_days,syms)
    missing_syms=[]
    missing_dates_bydate, missing_dates_bysym = {},{}
    for sym in syms:
        sdata = data[data.sym == sym.encode()]
        sdata = sdata[sdata.adjusted_close > 0 ]
        if sdata.shape[0]==0:
            missing_syms.append(sym)
            continue
        missing_days = set(trading_days)-set(sdata.date.dt.date)
        if missing_days:
            missing_dates_bysym[sym] = sorted(list(missing_days))
            for md in missing_days:
                if md not in missing_dates_bydate:
                    missing_dates_bydate[md]=[]
                missing_dates_bydate[md].append(sym)

    logging.info(f'There are {len(missing_syms)} missing symbols')
    if missing_dates_bydate:
        skeys = sorted(missing_dates_bydate.keys())
        for d in skeys:
            s = missing_dates_bydate[d]
            logging.info(f'Missing date {d}->{parr(s)}')
    if missing_dates_bysym:
        for s,d in missing_dates_bysym.items():
            logging.info(f'Missing sym {s}->{parr(d)}')
