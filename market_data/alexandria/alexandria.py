import numpy as np
import datetime
from utils.datagrid import CT, ColumnDef, DataGrid
from utils.qpthack import qconnection
import logging
from typing import List
import pandas as pd
class AlexandriaNewsSentiment(DataGrid):
    TF_BOOL = lambda x: True if x=='T' else False

    _SCHEMA = [
        ColumnDef('Timestamp',CT.DATETIME,isKey=True),
        ColumnDef('StoryID', CT.STR, isKey=True),
        ColumnDef('Source',CT.SYMBOL),
        ColumnDef('Ticker',CT.SYMBOL, isKey=True),
        ColumnDef('Country', CT.SYMBOL),
        ColumnDef('Isin', CT.SYMBOL),
        ColumnDef('Sentiment', CT.I8),
        ColumnDef('Confidence', CT.F32),
        ColumnDef('Novelty', CT.I64),
        ColumnDef('Events', CT.STR),
        ColumnDef('Relevance', CT.F32),
        ColumnDef('HeadlineOnly', CT.BOOL,converter=TF_BOOL),
        ColumnDef('AutoMessage', CT.BOOL,converter=TF_BOOL),
        ColumnDef('MarketImpactScore', CT.F32),
        ColumnDef('Prob_POS', CT.F32),
        ColumnDef('Prob_NTR', CT.F32),
        ColumnDef('Prob_NEG', CT.F32),
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.Timestamp.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(datetime.date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('alnews',self._SCHEMA)

    def load_df_to_kdb(self, fileName: str, qc: qconnection, KDB_ROOT: str):
        '''
            Load alexandria csv file to kdb
        :param fileName: location of the input file
        :param qc: qconnection
        :param KDB_ROOT: root director for kdb partition ( where sym file lives )
        '''

        self.kdbInitConnection(qc)
        chunk_size = 500000  # let's read 500000 rows at a time
        chunk = self.readCsvChunk(fileName=fileName, rows=chunk_size, startRow=0, sep='\t')
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
            chunk = self.readCsvChunk(fileName=fileName, rows=chunk_size, startRow=index * chunk_size + 1, sep='\t')

        if currentProcessedYear > 0:
            logging.info('Saving data to disk for year %s', currentProcessedYear)
            self.saveKdbTableToDisk(qc, currentProcessedYear, KDB_ROOT)

class AlexandriaNewsDailySummary(DataGrid):
    ''' Alexandria news sentiment rolled up by date and ticker'''

    _SCHEMA = [
        ColumnDef('date',CT.DATE,isKey=True),
        ColumnDef('TimePeriod',CT.SYMBOL,isKey=True),
        ColumnDef('Ticker',CT.SYMBOL, isKey=True),
        ColumnDef('Mentions', CT.LONG),
        ColumnDef('Sentiment', CT.F32),
        ColumnDef('Confidence', CT.F32),
        ColumnDef('Relevance', CT.F32),
        ColumnDef('MarketImpactScore', CT.F32),
        ColumnDef('Prob_POS', CT.F32),
        ColumnDef('Prob_NTR', CT.F32),
        ColumnDef('Prob_NEG', CT.F32),
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.date.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(datetime.date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('alnews_daily_summary',self._SCHEMA)

    def load_news_summary(self, qc: qconnection, KDB_ROOT, years : List[int]):
        '''
            Load news summary (by date/ticker) to kdb
        :param qc: qconnection
        :param KDB_ROOT: root director for kdb partition ( where sym file lives )
        :param years: list of years to load
        :return:
        '''
        self.kdbInitConnection(qc)
        for year in years:
            qcode = f'''
            {{[yr]            .t:update Confidence: ((Prob_POS|Prob_NTR|Prob_NEG) -(1%3)) % (2%3) from 
            select Mentions:`float$count i, avg Sentiment, avg Relevance, avg MarketImpactScore, avg Prob_POS, avg Prob_NTR, avg Prob_NEG 
            by LocalTimestamp.date,TimePeriod, Ticker 
            from 
            update LocalTimestamp: ltime Timestamp,
            TimePeriod:?[((`minute$ltime Timestamp)<09:30);`PREOPEN;?[((`minute$ltime Timestamp)<15:30);`CONTINUOUS;?[((`minute$ltime Timestamp)<16:00);`EOD;`POSTCLOSE]]]
            from 
            select from alnews where year = yr,(`year$(ltime Timestamp))=yr, Ticker <> `;
            0!.t
            }}
            '''

            data = self._sendSync(qc, qcode,year)
            data['date'] = data.date.astype('datetime64[ns]')  ##qpythin serliazation bug - cant handle np.datetime64[s]
            data['year'] = [year] * len(data)
            data['asof_date'] = [np.datetime64(datetime.date.today())] * len(data)
            data.meta = self.getqpythonMetaData()
            logging.info(f'Calculated {len(data)} rows for year {year}')
            self.kdbInitPartitionTable(year, qc)
            logging.info('Upserting %s rows for year %s', len(data), year)
            self.upsertToKDB(qc, KDB_ROOT, data)
            self.saveKdbTableToDisk(qc, year, KDB_ROOT)


    def get_all_sentiments(self,start_date: datetime.date, end_date: datetime.date, qc: qconnection, columns = []) -> pd.DataFrame:
        '''
            Get all news sentiment for a given date range
        :param start_date: start date
        :param end_date: end date
        :param qc: qconnection
        :return: pandas dataframe
        '''
        qcode = f'''
            {{[sd;ed] select {self.getColumnsWithCasts(columns)} from {self.name_} where date within (sd;ed)}}
            '''
        data = self._sendSync(qc, qcode, np.datetime64(start_date), np.datetime64(end_date))
        return self.castToPython(data)

    def get_sentiments_by_effective_date(self,start_date: datetime.date, end_date: datetime.date, qc: qconnection,
                                         sameDayPhases = ["PREOPEN","CONTINUOUS"], checkSentiment=True ) -> pd.DataFrame:
        '''
            Get a snapshot of news sentiment by date/ticker
        :param start_date: start date
        :param end_date: end date
        :param qc: qconnection
        :param sameDayPhases: list of day phases to include into same day ( e.g. PREOPEN, CONTINUOUS then POSTCLOSE moves to different day)

        :return: pandas dataframe
        '''
        checkSentimentQ = ',Sentiment <> 0' if checkSentiment else ''
        qcode = f'''
        {{
        [phases;sd;ed]
        0!update Confidence: ((Prob_POS|Prob_NTR|Prob_NEG) -(1%3)) % (2%3) from 
        select `float$(sum Mentions), Mentions wavg Sentiment,Mentions wavg Relevance,Mentions wavg MarketImpactScore,
        Mentions wavg Prob_POS,Mentions wavg Prob_NTR,Mentions wavg Prob_NEG
        by EffectiveDate,Ticker from
        update EffectiveDate:?[TimePeriod in `$phases;date;date+1] from
        select from alnews_daily_summary where date >=sd, date <= ed, Ticker <> ` {checkSentimentQ}
        }}
        '''

        data = self._sendSync(qc, qcode, sameDayPhases, np.datetime64(start_date), np.datetime64(end_date))
        return self.castToPython(data)
