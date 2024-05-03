import pandas as pd
import logging
from typing import List,Callable
from dataclasses import dataclass
from enum import Enum
from utils.qpthack import qconnection,MetaData, qtemporal

class CT(Enum):
    BOOL = 'bool','boolean', qtemporal.QBOOL_LIST
    DATE = 'datetime64[s]','date', qtemporal.QDATE_LIST
    DATETIME = 'datetime64[s]','datetime', qtemporal.QDATETIME_LIST
    F32 = 'float32','real', qtemporal.QFLOAT_LIST
    I64 =  'int64','long', qtemporal.QLONG_LIST
    I64WNA = 'Int64', 'long', qtemporal.QLONG_LIST
    I8 = 'int8','int', qtemporal.QINT_LIST
    STR = 'string','', qtemporal.QSTRING_LIST
    SYMBOL = 'string','sym', qtemporal.QSYMBOL_LIST
    LONG = 'Int32','long', qtemporal.QLONG_LIST

    def __init__(self, pandatype:str, kdbtype:str, qtemporalType:qtemporal = None):
        self.pandatype_ = pandatype
        self.kdbtype_ = kdbtype
        self.qtemporalType_ = qtemporalType

    @property
    def pandatype(self):
        return self.pandatype_
    @property
    def kdbtype(self):
        return self.kdbtype_

    @property
    def qtemportaltype(self):
        return self.qtemporalType_



@dataclass
class ColumnDef:
    name: str
    type: CT
    isKey: bool = False
    converter: Callable = None #converted is called while parsting file to pandas dataframe
    transformer: Callable = None #transformer is called after the file is read into pandas dataframe
    isPartition: bool = False #is this a kdb partition column
    csvOrigName: str = None #name of the column in the csv file if different from the name in the dataframe
    def isDate(self):
        return self.type in [CT.DATE,CT.DATETIME]
    @property
    def csvName(self):
        return self.csvOrigName if self.csvOrigName is not None else self.name


class DataGrid:
    ''' Class that abstracts between pandas dataframe and partitioned kdb table
    '''
    def __init__(self,name:str,columns:List[ColumnDef]):
        self.name_ = name
        self.columns_ = columns

    def columnNameMapper(self):
        ''' Remapper of names from dataframe to what we want them called '''
        mapper = {}
        for c in self.columns_:
            if c.csvOrigName is not None and c.csvOrigName != c.name:
                mapper[c.csvOrigName] = c.name
        return mapper

    @staticmethod
    def _getKDBtype(c:ColumnDef):
        return c.type.kdbtype
    def readCsvChunk(self,fileName:str,rows:int,startRow:int=0, sep:str='\t', usecols=None)->pd.DataFrame:
        dtypes = {c.csvName:c.type.pandatype for c in self.columns_ if c.converter is None and not c.isDate()}
        converters = {c.csvName:c.converter for c in self.columns_ if c.converter is not None}
        datecols = [c.csvName for c in self.columns_ if c.isDate() and not c.transformer]
        df = pd.read_csv(fileName, sep=sep,dtype=dtypes,converters=converters,parse_dates=datecols,nrows=rows,skiprows=range(1,int(startRow)), usecols=usecols)
        for c in self.columns_:
            if c.transformer is not None:
                if len(df) > 0:
                    df[c.name] = c.transformer(df)
                else:
                    df[c.name] = [] ## for schema consistency sake
        mapper = self.columnNameMapper()
        if mapper and len(mapper) > 0:
            df.rename(columns=mapper,inplace=True)
        return df

    def getKeyColumns(self) -> List[ColumnDef]:
        return [c for c in self.columns_ if c.isKey]

    def getColumnOrder(self) -> List[str]:
        return [c.name for c in self.columns_ if c.isKey] + [c.name for c in self.columns_ if not c.isKey]
    def getPartitionColumn(self) -> ColumnDef:
        pcols = [c for c in self.columns_ if c.isPartition]
        assert ( 1 == len(pcols))
        return pcols[0]

    def getqpythonMetaData(self) -> MetaData:
        meta = MetaData()
        for c in self.columns_:
            meta[c.name] = c.type.qtemportaltype
        return meta

    def _sendSync(self, qconnection, qcode, *parameters):
        logging.warning(f'EXECUTING {qcode}')
        return qconnection.sendSync(qcode, *parameters)
    def kdbInitConnection(self,qconnection:qconnection):
        ''' Initialize the provided connection by creating the table if it does not alread exist'''
        nl = '\n'
        def ti(c) :
            return '`$()' if c.type == CT.SYMBOL else f'`{c.type.kdbtype}$()' if len(c.type.kdbtype) > 0 else "()"
        keyColDef = ';'.join([f' {c.name} : {ti(c)}' for c in self.getKeyColumns()])
        nonKeyColDef = ';'.join([f' {c.name} : {ti(c)} ' for c in self.columns_ if not c.isKey])
        tableInitCode = f"{self.name_}:([{keyColDef}] {nl} {nonKeyColDef} {nl} );"
        qcode = f"if [not `{self.name_} in key `.;{nl}{tableInitCode}{nl} ];"
        self._sendSync(qconnection,qcode)

    def upsertToKDB(self, qconnection:qconnection, KDB_ROOT:str, df:pd.DataFrame):
        '''
            Upsert the provided dataframe to the kdb table
            @param qconnection: qconnection object
            @param KDB_ROOT: root directory for the kdb partition ( which contains sym file )
            @param df: dataframe to upsert
        '''
        partitions = sorted(list(df[self.getPartitionColumn().name].unique()))

        assert len(partitions) == 1
        dataChunk = self._fixQPythonNulls(df[self.getColumnOrder()])
        dataChunk.meta = self.getqpythonMetaData()  # special member variable that qpython looks at
        qcode = f'{{[d] KDB_ROOT:hsym `$"{KDB_ROOT}";`{self.name_}_upd upsert .Q.en[KDB_ROOT; 0!d];}}'
        self._sendSync(qconnection,qcode, dataChunk)
    def _fixQPythonNulls(self,df:pd.DataFrame) ->pd.DataFrame:
        ''' Workaround qpython bug in handling nulls in string columns'''
        dfClean = df.copy()
        for c in self.columns_:
            if c.type.qtemporalType_ in [qtemporal.QSTRING_LIST,qtemporal.QSYMBOL_LIST] and c.name in dfClean.columns:
                dfClean[c.name].fillna('',inplace=True)
        return dfClean
    def saveKdbTableToDisk(self, qconnection:qconnection, partitionValue, KDB_ROOT:str):
        '''
            Save the kdb table to disk
            @param qconnection: qconnection object
            @param partitionValue: partition value (e.g. year) to save
            @param KDB_ROOT: root directory for the kdb partition ( which contains sym file )
        '''
        qcode = f''' 
            KDB_ROOT:"{KDB_ROOT}";
            TABLEROOT: hsym `$(KDB_ROOT,"{partitionValue}/","{self.name_}/");
            TABLEROOT set .Q.en[hsym `$KDB_ROOT] 0!{self.name_}_upd;
        '''
        self._sendSync(qconnection,qcode)
    def kdbInitPartitionTable(self, partitionValue,qconnection:qconnection):
        '''
            Create an intermediate table corresponding to a single partition which we will upsert data to

            N.B Intermediate table ( _upd below ) is required because kdb partitioned table cannot be keyed
            Thus we upsert into an intermediate (keyed) table and then copy data from that table to the partition
        '''
        pcol = self.getPartitionColumn()
        qcode = f'{self.name_}_upd: {"".join(["`"+c.name for c in self.getKeyColumns()])} xkey ' \
                f'(select from {self.name_} where {pcol.name} = {partitionValue});'
        self._sendSync(qconnection,qcode)

    def getColumnsWithCasts(self, columns = [] ) -> str:
        ''' Workaround qpython bug in serializing ints coming back from kdb.
            We cast all ints to float to avoid this problem
        '''
        typedCols = []
        for c in self.columns_:
            if columns is None or len(columns) == 0 or c.name in columns:
                typedCols.append(f'`float${c.name}' if c.type in [CT.I64,CT.I8, CT.LONG] else c.name)
        return ', '.join(typedCols)

    def castToPython(self, df: pd.DataFrame):
        ''' Cast the dataframe to python types to avoid qpython serialization bug'''
        df = df.copy()
        for c in self.columns_:
            if c.name in df.columns:
                if c.type in [CT.I64,CT.I8, CT.LONG, CT.SYMBOL]:
                    df[c.name] = df[c.name].astype(c.type.pandatype)
                elif c.type == CT.DATE:
                    df[c.name] = df[c.name].dt.date
                elif c.type == CT.DATETIME:
                    df[c.name] = df[c.name].dt.to_pydatetime()
        return df
