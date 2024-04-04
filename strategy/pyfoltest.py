
from utils.qpthack import qconnection, qtemporal, MetaData
import pandas as pd
from market_data.alexandria.alexandria import AlexandriaNewsDailySummary
from market_data.market_data import get_md_conn
from strategy import ARIMASignal, AvelanedaSignal
from datetime import date, timedelta
from utils.utils import cached_df, get_next_trading_days
from quant.symreturns import SymReturns
from strategy.ensemble import getAllSignals
import pyfolio as pf
import numpy as np

if __name__ == '__main__':
    signals = getAllSignals(date(2007,1,1),date(2023,12,31))
    signals.set_index('date', inplace=True, drop=False)
    signals['avel_direction'] = 0
    signals.loc[signals.signal < 0, 'avel_direction'] = 1
    signals.loc[signals.signal > 0, 'avel_direction'] = -1
    MEASURES = sorted(['c2c', 'o2c', 'c2cdn', 'o2cdn', 'c2cbn', 'o2cbn', 'c2csn', 'o2csn'])

    # Drop bad prices - mostly from news referring to GOOG vs GOOGL, BRKB vs -B etc.
    signals.replace([np.inf, -np.inf], np.nan, inplace=True)
    signals.dropna(subset=MEASURES, inplace=True)  # Mostly news referring to GOOG vs GOOGL, BRKB vs -B etc.

    dailyPos = signals[signals.avel_direction != 0][MEASURES].mul(signals.avel_direction[signals.avel_direction!=0], axis=0)
    returns = dailyPos.groupby(dailyPos.index).mean()

    benchmarkReturns = SymReturns().getReturns(returns.index.min(),returns.index.max(),['SPY'])
    benchmarkReturns.index = pd.to_datetime(benchmarkReturns.date)

    returns.index = pd.to_datetime(returns.index) #expects pd.DatetimeIndex
    ts = pf.create_full_tear_sheet(returns = returns.o2cbn,benchmark_rets=benchmarkReturns.o2cbn)
    print(ts)
