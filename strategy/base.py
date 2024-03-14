from dataclasses import dataclass
from quant.symreturns import SymReturns
from symbology.compositions import get_composition
from utils.utils import get_trading_days,get_last_trading_month_end
import logging
from datetime import  date
from typing import List
from dateutil.relativedelta import relativedelta
from utils.datagrid import ColumnDef, DataGrid
from market_data.market_data import get_md_conn, _kdbdt
@dataclass
class TraderParams:
    ''' Base class for strategy parameters'''
    universe:str # universe to consider
    target: str # target variable 'c2c', 'c2cdn', 'c2cbn', 'c2csn' etc
    ## return cleanup
    minPrice : float = 1.0 # minimum price of stock to consider
    meanMaxSD : float = 3.0 # maximum mean return in SDs from average to include
    volMaxSd : float = 6.0 # maximum vol in SDs from average to include
    ## training parameters
    trainPeriod: relativedelta = relativedelta(months=6) # period to train the model
    trainFrequency: relativedelta = relativedelta(months=1) # frequency of training

class Trader:
    ''' Base srategy '''
    def __init__(self,params : TraderParams):
        self.params_ = params

    def getCleanData(self,startDate:date,endDate:date, symbols:List[str] = None, removeOutliers = True):
        ''' Get returns for given period '''
        if symbols is None:
            comp = get_composition(self.params_.universe, get_last_trading_month_end(endDate))
            symbols = sorted(list(comp.ticker.unique()))
        else:
            symbols = sorted(symbols)
        returns = SymReturns().getReturns(startDate,endDate,symbols)
        ##Remove syms where some of the returns could not be computed ( not enough history, etc )
        original_syms = set(returns.sym.unique())
        returns = returns.dropna()
        excluded_syms = original_syms - set(returns.sym.unique())
        if excluded_syms:
            logging.info(f'Excluded {len(excluded_syms)} symbols with NULL data: {",".join(sorted(excluded_syms))} ')

        ##Remove all data for which we do not have the full period
        trading_days = get_trading_days('NYSE', startDate, endDate)
        original_syms = set(returns.sym.unique())
        returns = returns.groupby('sym').filter(lambda x: x.shape[0] == len(trading_days))
        # log symbols that were excluded
        excluded_syms = original_syms - set(returns.sym.unique())
        if excluded_syms:
            logging.info(f'Excluded {len(excluded_syms)} symbols with missing dates: {",".join(sorted(excluded_syms))} ')

        ##Remove outliers - syms that either experiences large directional or vol moves during the period
        if removeOutliers:
            if self.params_.minPrice is not None :
                ##remove "penny" stocks - stocks where price fell below threshold during the period
                # remove "penny stocks" - stocks whose price dips below 1$ ever
                original_syms = set(returns.sym.unique())
                returns = returns.groupby('sym').filter(lambda x: x.close.min() > 1.)
                excluded_syms = original_syms - set(returns.sym.unique())
                if excluded_syms:
                    logging.info(f'Excluded {len(excluded_syms)} symbols with penny prices: {",".join(sorted(excluded_syms))} ')

            if self.params_.meanMaxSD is not None or self.params_.volMaxSd is not None:
                sym_stats = returns.groupby(['sym'], as_index=True)['c2c'].agg(['mean', 'std'])
                outliers = set()
                if self.params_.meanMaxSD is not None:
                    outliers = outliers.union(sym_stats[sym_stats["mean"].abs() > self.params_.meanMaxSD * sym_stats["mean"].std()].index)
                if self.params_.volMaxSd is not None:
                    outliers = outliers.union(sym_stats[sym_stats["std"] > self.params_.volMaxSd * sym_stats["std"].std()].index)
                if outliers:
                    logging.info(f'Excluded {len(outliers)} symbols with outlier returns: {",".join(sorted(outliers))} ')
                returns = returns[~returns.sym.isin(outliers)]
        returns.sort_values(['sym', 'date'], inplace=True)
        return returns

class TradingSignal(DataGrid):
    '''Represents signal computed by some Trader strategy'''
    def __init__(self, tableName:str, schema:List[ColumnDef]):
        super().__init__(tableName, schema)

    def retrieveSignal(self, startDate:date, endDate:date, syms:List[str] = None, columns:List[str] = [], additionalWhereClause:str = None):
        '''Retrieve signal for given period'''
        if startDate is None and endDate is None and syms is None:
            raise Exception("Must specify a where condition")
        startDate,endDate = startDate or date.min,endDate or date.max
        whereClause = f'where year >= `year$sd, year <=`year$ed,date>=sd,date<=ed'
        if syms is not None:
            whereClause += ',sym in `$syms'
        if additionalWhereClause is not None:
            whereClause += ',' + additionalWhereClause
        query = f'''
            {{[syms;sd;ed]
                (select {self.getColumnsWithCasts(columns)} from {self.name_} {whereClause})
            }}
        '''
        with get_md_conn() as q:
            data =  self._sendSync(q,query,syms or ['ALL'], _kdbdt(startDate),_kdbdt(endDate))
            return self.castToPython(data)
