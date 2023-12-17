from datetime import date
from symbology.compositions import get_composition_history, get_broad_etfs, get_constituent_history
from utils.qpthack import qconnection,MetaData, qtemporal
import logging
import yfinance as yf
import pandas as pd
import numpy as np


logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')

UNDERLYING_ETFS = "IWB IWM IWV IVV".split()
START_DATE = date(2000,1,1)
END_DATE = date(2023,9,1)
BATCH_SIZE = 500
ASOF_DATE = date.today()
universe = set()
for e in UNDERLYING_ETFS:
    comp_history = get_composition_history(e,START_DATE,END_DATE)
    universe = universe.union(set(comp_history.ticker.unique().tolist()))
universe = sorted(list(universe.union(set(get_broad_etfs()))))
logging.info(f"Got {len(universe)} symbols")

batches = [universe[offs : offs+BATCH_SIZE] for offs in range(0, len(universe), BATCH_SIZE)]

#TEMP DEBUG ONLY
##batches = [batches[10]]

INIT_Q = '''
DB_PATH:":c:/KDB_MARKET_DATA/";
if [ not `daily_prices in key `.;
    $ [ `DAILY_PRICES in key `$DB_PATH:
        daily_prices: 2!(get `$(DB_PATH,"DAILY_PRICES"));
        daily_prices:
        ([ date:`date$();Ticker:`symbol$() ] Open:`real$();High:`real$();Low:`real$();Close:`real$();
            Volume:`long$();Dividends:`real$();StockSplits:`real$();asof_date:`date$())
    ];
];
'''

with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
    q.sendSync(INIT_Q)

for index,batch in enumerate(batches):
    logging.info(f"Processing batch {index}/{len(batches)}")
    md = yf.download(batch,start=START_DATE.strftime('%Y-%m-%d'),end=END_DATE.strftime('%Y-%m-%d'),
                              actions=True,auto_adjust=True,progress=True,repair=True,group_by='Ticker')
    md = md.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    md['date']=pd.to_datetime(md.index)
    md['asof_date']=pd.to_datetime(ASOF_DATE)
    if 'Capital Gains' in md.columns:
        md.drop(columns='Capital Gains',inplace=True)
    md.rename(columns={'Stock Splits':'StockSplits'},inplace=True)
    md = md['date Ticker Open High Low Close Volume Dividends StockSplits asof_date'.split()]

    md.meta = MetaData(asof_date=qtemporal.QDATE,date=qtemporal.QDATE,Volume=qtemporal.QLONG)
    for c in 'Open High Low Close StockSplits Dividends'.split():
        md.meta[c]=qtemporal.QFLOAT

    with qconnection.QConnection(host='localhost',port=12345,pandas=True) as q:
        out=q.sendSync('{[x] `daily_prices upsert x; select SUMC:sum Close,AVGC:avg Close from x}',md)
        if (out['SUMC'][0] - md.Close.sum())/out['SUMC'][0] > 1e-5:
            logging.error(f"DATA LOSS FOR SUM CHECK {out['SUMC'][0]} vs {md.Close.sum()}")
            raise Exception(f"DATA LOSS FOR SUM CHECK {out['SUMC'][0]} vs {md.Close.sum()}")
        if (out['AVGC'][0] - md.Close.mean()) / out['AVGC'][0] > 1e-5:
            logging.error(f"DATA LOSS FOR AVG CHECK {out['AVGC'][0]} vs {md.Close.mean()}")
            raise Exception(f"DATA LOSS FOR AVG CHECK {out['AVGC'][0]} vs {md.Close.mean()}")

        #print(q.sendSync('{[x] meta x}',md))


##Persist to DISK
PERSIST_Q = '''
    MDROOT:"c:/KDB_MARKET_DATA";
    TABLEROOT: hsym `$(MDROOT,"/daily_prices/");
    TABLEROOT set .Q.en[hsym `$MDROOT] 0!daily_prices
'''
with qconnection.QConnection(host='localhost', port=12345, pandas=True) as q:
    q.sendSync(PERSIST_Q)
