import logging

import pandas as pd

from strategy.ARIMATrader import ARIMASignal
from strategy.AvelanedTrader import AvelanedaSignal
from datetime import date
import pandas as pd
import numpy as np
from quant.symreturns import SymReturns
from quant.symstats import SymStats

def computeOverlap(sd:date,ed:date,avel_cutoffs=(-1.5,1.5), measures = ['c2c','o2c','c2cbn','o2cbn','c2cdn','o2cdn','c2csn','o2csn']):
    '''Compute overlap between ARIMA and Avelaneda strategies'''
    arimaS = ARIMASignal().retrieveSignal(sd,ed)
    avelS = AvelanedaSignal().retrieveSignal(sd,ed)
    avelS = avelS[(avelS.signal<avel_cutoffs[0]) | (avelS.signal>avel_cutoffs[1])]
    merged = pd.merge(arimaS,avelS,on=['sym','date'],suffixes=('_arima','_avel'),how='outer')

    merged['arima_signal'] = 'NONE'
    merged['avel_signal'] = 'NONE'
    merged.loc[(~merged.prediction.isna()) & (np.sign(merged.prediction)  > 0), 'arima_signal'] = 'BUY'
    merged.loc[(~merged.prediction.isna()) & (np.sign(merged.prediction) < 0) , 'arima_signal'] = 'SELL'
    merged.loc[(~merged.signal.isna()) & ( merged.signal < avel_cutoffs[0]), 'avel_signal'] = 'BUY'
    merged.loc[(~merged.signal.isna()) &( merged.signal > avel_cutoffs[1]), 'avel_signal'] = 'SELL'
    merged = SymReturns().enrichWithReturns(merged,columns=['sym','date']+measures)
    merged = SymStats().enrichWithSymStats(merged, ['sym', 'date', 'beta', 'etf', 'volatility', 'ADVD'])

    aggrules = { x: 'mean' for x in measures}
    aggrules['sym'] = 'count'
    dailyStats = merged.groupby(['date','arima_signal','avel_signal']).agg(aggrules).reset_index(drop=False).rename(columns={'sym':'trades'})
    return dailyStats,merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    stats, positions = [], []
    measures = sorted(['c2c', 'o2c', 'c2cbn', 'o2cbn', 'c2cdn', 'o2cdn', 'c2csn', 'o2csn'])
    for year in range(2007,2024):
        logging.info(f'Computing overlap for year {year}')
        yStats, yPositions = computeOverlap(date(year,1,1),date(year,12,31), measures=measures)
        stats.append(yStats)
        positions.append(yPositions)
    stats = pd.concat(stats)

    positions = pd.concat(positions)
    positions.to_pickle("C:/TEMP/positions_3_13.gz")
    logging.info("Saved positions")
    stats.to_pickle("C:/TEMP/stats_3_13.gz")
    logging.info("Saved stats")

    aggrules = {x: 'mean' for x in measures}
    aggrules['trades'] = 'sum'
    totalStats = stats.groupby(['arima_signal','avel_signal']).agg(aggrules).reset_index(drop=False)
    ##bps
    for c in measures:
        totalStats[c] = totalStats[c]*10000
    print(totalStats)