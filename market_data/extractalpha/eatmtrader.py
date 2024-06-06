from dataclasses import dataclass
from typing import List, Optional

from dateutil.relativedelta import relativedelta

from market_data.extractalpha.datasets import EATM, EACAM
from market_data.market_data import get_md_conn
from quant.symreturns import SymReturns
from datetime import date
import pandas as pd
import logging
import numpy as np

from utils.datagrid import ColumnDef, CT
from strategy.ensemble import filterByUniverse
from strategy.base import Trader, TraderParams, TradingSignal


@dataclass
class EATMTraderParams(TraderParams):
    SELL_SCORE_THRESHOLD: int = 20  # Anything <= 20 is a sell signal
    BUY_SCORE_THRESHOLD: int = 80  # Anything >= 80 is a buy signal
    minPrice: float = 4.0
    minVolume: float = 10000.0
    model: str = 'TM'  # TM or CAM


DEFAULT_EATM_PARAMS = EATMTraderParams(target='o2cbn', universe='IWV')


class EATMSignal(TradingSignal):
    _SCHEMA = [
        ColumnDef('sym', CT.SYMBOL, isKey=True),
        ColumnDef('date', CT.DATE, isKey=True),
        ColumnDef('score', CT.I8),  # 0-100
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(date.today())] * len(frame)),
        ColumnDef('year', CT.LONG, transformer=lambda frame: frame.date.dt.year, isPartition=True),
    ]

    def __init__(self):
        super().__init__('eatm_signal', self._SCHEMA)


class EATMTrader(Trader):
    """ EA TM Model based strategy """

    def __init__(self, params: EATMTraderParams = DEFAULT_EATM_PARAMS):
        super().__init__(params)
        self.params_ = params
        self.modelH_ = EATM() if params.model == 'TM' else EACAM()
        self.sr_ = SymReturns()

    def getModelScores(self, startDate: date, endDate: date, syms: List[str] = None):
        scoreCol = 'TM1' if self.params_.model == 'TM' else 'CAM1'
        columns = 'year Date Ticker'.split() + [scoreCol]
        whereClause = f' (({scoreCol} >= {self.params_.BUY_SCORE_THRESHOLD}) | ({scoreCol} <= {self.params_.SELL_SCORE_THRESHOLD}))'
        tmdata = self.modelH_.retrieveSignal(startDate, endDate, syms=syms, columns=columns,
                                             additionalWhereClause=whereClause, symColumnName='Ticker')
        if tmdata is None or not len(tmdata):
            return None
        tmdata.rename(columns={'Date': 'date', 'Ticker': 'sym'}, inplace=True)
        tmdata = filterByUniverse(tmdata, universe=self.params_.universe)
        tmdata = self.sr_.enrichWithReturns(tmdata, columns=['sym', 'date', 'o2c', 'o2cdn', 'o2csn', 'o2cbn'])
        tmdata = tmdata[(tmdata.volume > self.params_.minVolume) & (tmdata.close >= self.params_.minPrice)]
        tmdata = tmdata.replace([np.inf, -np.inf], np.nan).dropna()
        tmdata.rename(columns={scoreCol: 'score'}, inplace=True)
        return tmdata

    def getActionableUniverse(self, startDate: date, endDate: date) -> Optional[List[str]]:
        """
            Get actionable universe for a given time period
            Returns a list of symbols that the model performed well on during the period
        """
        modelScores = self.getModelScores(startDate, endDate)
        if modelScores is None or not len(modelScores):
            logging.warning(f'No data for period {startDate} - {startDate}')
            return None
        sideMultiplier = 2 * (modelScores.score > 50).astype(int) - 1
        sideAdjustedReturns = sideMultiplier * modelScores[self.params_.target]
        bySym = sideAdjustedReturns.groupby(modelScores.sym).mean()
        actionableUniverse = sorted(
            bySym[bySym > 0].index.tolist())  # symbols that have positive returns during training
        return actionableUniverse


def calcAndStoreSignals(startDate: date, endDate: date, DRY_RUN=True):
    # initialize KDB connection and temp tables
    if not DRY_RUN:
        signal = EATMSignal()
        KDB_ROOT = 'C://KDB_MARKET_DATA2/'
        with get_md_conn() as q:
            signal.kdbInitConnection(q)

    params = DEFAULT_EATM_PARAMS
    trader = EATMTrader(params)

    SIM_DATES = []
    SIM_DATE = startDate
    while SIM_DATE < endDate:
        SIM_DATES.append(SIM_DATE)
        SIM_DATE = SIM_DATE + params.trainFrequency
    currentProcessedYear = -1
    for index, SIM_DATE in enumerate(SIM_DATES):
        TRAIN_PERIOD_START = SIM_DATE - params.trainPeriod
        TRAIN_PERIOD_END = SIM_DATE - relativedelta(days=1)
        TEST_PERIOD_START = SIM_DATE
        TEST_PERIOD_END = SIM_DATE + params.trainFrequency

        logging.info(
            f'{index}/{len(SIM_DATES)} TRAINING {TRAIN_PERIOD_START} - {TRAIN_PERIOD_END} TESTING {TEST_PERIOD_START} - {TEST_PERIOD_END}')

        actionableUniverse = trader.getActionableUniverse(TRAIN_PERIOD_START, TRAIN_PERIOD_END)
        if actionableUniverse is None or not len(actionableUniverse):
            continue
        testModelScores = trader.getModelScores(TEST_PERIOD_START, TEST_PERIOD_END, syms=actionableUniverse)
        if testModelScores is None or not len(testModelScores):
            continue
        testModelScores['asof_date'] = np.datetime64(date.today())
        testModelScores['date'] = pd.to_datetime(testModelScores.date)
        testModelScores['year'] = testModelScores.date.dt.year

        years = sorted(list(testModelScores.year.unique()))

        if not DRY_RUN:
            with get_md_conn() as q:
                for year in years:
                    if year > currentProcessedYear:
                        if currentProcessedYear > 0:
                            logging.info(f'Saving data to disk for year {currentProcessedYear}')
                            signal.saveKdbTableToDisk(q, currentProcessedYear, KDB_ROOT)
                        currentProcessedYear = year
                        logging.info(f'Initializing partition table for year {year}')
                        signal.kdbInitPartitionTable(year, q)

                    data = testModelScores[testModelScores.year == year].copy()
                    data.meta = signal.getqpythonMetaData()
                    signal.upsertToKDB(q, KDB_ROOT, data)

                if currentProcessedYear > 0:
                    signal.saveKdbTableToDisk(q, year, KDB_ROOT)


if __name__ == '__main__UNCOMMENTME':
    logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    calcAndStoreSignals(date(2001, 7, 1), date(2023, 12, 29), DRY_RUN=False)
    # .Q.chk hsym `$"C:/KDB_MARKET_DATA2/"
