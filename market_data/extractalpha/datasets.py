from utils.qpthack import qconnection,qtemporal
import numpy as np
import datetime
from utils.datagrid import CT, ColumnDef
from strategy.base import TradingSignal
import logging

#TM_FILE = r'C:\Users\orduk\OneDrive\Documents\ExtractAlpha\TM1_History_2000_202312.zip'
#CAM_FILE = r'C:\Users\orduk\OneDrive\Documents\ExtractAlpha\CAM1_History_2005_202312.zip'

class BaseEADataset(TradingSignal):
    ''' Base class for ExtractAlpha datasets '''
    def __init__(self,name:str,columns):
        super().__init__(name,columns)
    def load_df_to_kdb(self, fileName: str, qc: qconnection, KDB_ROOT: str):
        '''
            Load  csv file to kdb
        :param fileName: location of the input file
        :param qc: qconnection
        :param KDB_ROOT: root director for kdb partition ( where sym file lives )
        '''

        self.kdbInitConnection(qc)
        chunk_size = 500000  # let's read 500000 rows at a time
        chunk = self.readCsvChunk(fileName=fileName, rows=chunk_size, startRow=0,sep=',')
        index = 0
        currentProcessedYear = -1
        while (chunk is not None and len(chunk) > 0):
            years = sorted(list(chunk.year.unique()))
            for year in years:
                if year > currentProcessedYear:
                    if currentProcessedYear > 0:
                        logging.info('Saving data to disk for year %s', currentProcessedYear)
                        self.saveKdbTableToDisk(qc, currentProcessedYear, KDB_ROOT)
                    currentProcessedYear = year
                    logging.info('Initializing partition table for year %s', year)
                    self.kdbInitPartitionTable(year,qc)
                df = chunk[chunk.year == year]
                df.meta = self.getqpythonMetaData()
                logging.info('Upserting %s rows for year %s', len(df), year)
                self.upsertToKDB(qc, KDB_ROOT, df)
            maxYear = chunk.year.max()
            index += 1
            chunk = self.readCsvChunk(fileName=fileName, rows=chunk_size, startRow=index * chunk_size + 1, sep=',')

        if currentProcessedYear > 0:
            logging.info('Saving data to disk for year %s', currentProcessedYear)
            self.saveKdbTableToDisk(qc, currentProcessedYear, KDB_ROOT)

class EATM(BaseEADataset):
    ''' ExtractAlpha Tactical (TM1) model - technical signal from reversion/factor momentum /liquidity shock/seasonality'''
    _SCHEMA = [
        ColumnDef('Date',CT.DATE,isKey=True),
        ColumnDef('Ticker', CT.SYMBOL, isKey=True,csvOrigName='Ticker_PointInTime'),
        ColumnDef('CUSIP', CT.SYMBOL, isKey=True, csvOrigName='CUSIP_PointInTime'),
        ColumnDef('Ticker_NOW', CT.SYMBOL, isKey=False,csvOrigName='Ticker'),
        ColumnDef('CUSIP_NOW', CT.SYMBOL, isKey=False,csvOrigName='CUSIP'),
        ColumnDef('Reversal_Component',CT.I64),
        ColumnDef('Factor_Momentum_Component', CT.I64),
        ColumnDef('Liquidity_Shock_Component', CT.I64),
        ColumnDef('Seasonality_Component', CT.I64),
        ColumnDef('TM1',CT.I64),

        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.Date.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(datetime.date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('eatm',self._SCHEMA)


class EACAM(BaseEADataset):
    ''' ExtractAlpha Cross-Asset (CAM1) model - stock signal from options market'''
    _SCHEMA = [
        ColumnDef('Date',CT.DATE,isKey=True),
        ColumnDef('Ticker', CT.SYMBOL, isKey=True,csvOrigName='Ticker_PointInTime'),
        ColumnDef('CUSIP', CT.SYMBOL, isKey=True, csvOrigName='CUSIP_PointInTime'),
        ColumnDef('Ticker_NOW', CT.SYMBOL, isKey=False,csvOrigName='Ticker'),
        ColumnDef('CUSIP_NOW', CT.SYMBOL, isKey=False,csvOrigName='CUSIP'),
        ColumnDef('Spread_Component',CT.I64WNA),
        ColumnDef('Skew_Component', CT.I64WNA),
        ColumnDef('Volume_Component', CT.I64WNA),
        ColumnDef('CAM1',CT.I64WNA),
        ColumnDef('CAM1_Slow', CT.I64WNA),
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.Date.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(datetime.date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('eacam',self._SCHEMA)

