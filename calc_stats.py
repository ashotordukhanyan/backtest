from market_data.market_data import get_all_syms,get_md_conn
import logging
import numpy as np
from utils.qpthack import qconnection,qtemporal,MetaData
import pandas as pd
from datetime import date
logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')
def persist_stats():
    ASOF_DATE = date.today()

    INIT_Q = '''
    DB_PATH:":c:/KDB_MARKET_DATA/";
    if [ not `daily_stats in key `.;
            daily_stats:
            ([ date:`date$();Ticker:`symbol$() ] LogRet:`real$();AnnualVol:`real$();ADV:`real$();asof_date:`date$())
    ];
    '''
    with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
        q.sendSync(INIT_Q)

    BATCH_SIZE=100


    universe = get_all_syms()
    batches = [universe[offs : offs+BATCH_SIZE] for offs in range(0, len(universe), BATCH_SIZE)]
    for index,batch in enumerate(batches):
        logging.info(f'Processing batch {index}/{len(batches)}')
        with get_md_conn() as q:
            md = q.sendSync('{[syms] select date,Ticker,`float$Volume,Close from daily_prices where Ticker in `$syms}',batch)
        #NB cast to float needed because returning int from kdb triggers some qpython pandas integration bug
        md['Ticker'] = md.Ticker.str.decode('utf-8')
        #md.set_index(['Ticker', 'date'], drop=True, inplace=True)
        md.sort_values(['Ticker', 'date'],inplace=True)
        md.loc[md['Ticker'] == md['Ticker'].shift(), 'LogRet'] = \
            np.log(md['Close']) - np.log(md['Close'].shift())
        md['AnnualVol'] = md.groupby('Ticker')['LogRet'].apply(lambda x: x.rolling(252,min_periods=252).std()).reset_index(level=0,drop=True)
        md['ADV'] = md.groupby('Ticker')['Volume'].apply(
            lambda x: x.rolling(21, min_periods=21).median()).reset_index(level=0, drop=True)
        md['asof_date'] = pd.to_datetime(ASOF_DATE)
        toSave = md['date Ticker LogRet AnnualVol ADV asof_date'.split()].reset_index(drop=True)
        toSave.meta = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE)
        for c in 'LogRet AnnualVol ADV'.split():
            toSave.meta[c]=qtemporal.QFLOAT

        with qconnection.QConnection(host='localhost',port=12345,pandas=True) as q:
            q.sendSync('{[x] `daily_stats upsert x}',toSave)
            #print(q.sendSync('{[x] meta x}',toSave))

    ##Persist to DISK
    PERSIST_Q = '''
        MDROOT:"c:/KDB_MARKET_DATA";
        TABLEROOT: hsym `$(MDROOT,"/daily_stats/");
        TABLEROOT set .Q.en[hsym `$MDROOT] 0!daily_stats
    '''
    with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
        q.sendSync(PERSIST_Q)


if __name__  == '__main__':
    persist_stats()