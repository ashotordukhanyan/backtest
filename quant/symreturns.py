import pandas as pd
from market_data.market_data import get_all_syms,get_md_conn,_kdbdt
import logging
from utils.datagrid import CT, ColumnDef, DataGrid
import numpy as np
from quant.symstat_params import SYMSTAT_PARAMS as PARAMS
from datetime import date,timedelta
def calculate_rolling_betas(mrets: pd.DataFrame ):
    '''
        Given a frame of monthly returns, calculate rolling betas to SPY,
        most correlated ETF and corresponding etf beta for each stock
    '''
    # Calculate SPY betas
    var_spy = mrets['SPY'].rolling(window=PARAMS.BETA_WINDOW_SIZE, min_periods=PARAMS.BETA_MIN_OBS).var()
    #matrix where each col is rolling cov of stock with SPY
    cov_with_spy = mrets.rolling(window=PARAMS.BETA_WINDOW_SIZE,min_periods=PARAMS.BETA_MIN_OBS)\
        .cov(mrets['SPY'])
    market_betas = cov_with_spy.div(var_spy, axis=0).clip(PARAMS.BETA_LIMITS[0],PARAMS.BETA_LIMITS[1])

    # Now calculate highest correlated ETF for each stock
    ETFS = sorted([x for x in PARAMS.FACTOR_ETFS if x in mrets.columns]) ## Not all ETFs are present in every time period
    correlations = mrets.rolling(window=PARAMS.BETA_WINDOW_SIZE, min_periods=PARAMS.BETA_MIN_OBS).corr(
        other=mrets[ETFS], pairwise=True)
    correlated_etfs = correlations.groupby('date').idxmax().applymap(lambda x: x[1] if isinstance(x,tuple) else x)
    correlated_etf_correlation = correlations.groupby('date').max()

    #Now calculated betas vs most correlated ETF
    var_etfs = mrets[ETFS].rolling(window=PARAMS.BETA_WINDOW_SIZE,
                                                                 min_periods=PARAMS.BETA_MIN_OBS).var()
    var_etfs_unstacked = var_etfs.unstack().reset_index().rename(columns={'sym':'etf',0: 'etf_var'})

    cov_with_etfs = mrets.rolling(window=PARAMS.BETA_WINDOW_SIZE,min_periods=PARAMS.BETA_MIN_OBS)\
        .cov(mrets[ETFS], pairwise=True)
    cov_with_etfs.index.set_names(['date','etf'],inplace=True)
    cov_with_etfs_unstacked = cov_with_etfs.unstack().unstack().reset_index().rename(columns= {0:'etf_cov'})

    result = pd.DataFrame({'beta': market_betas.unstack(), 'etf': correlated_etfs.unstack(), 'etf_corr': correlated_etf_correlation.unstack()}).reset_index()
    result = result.merge(cov_with_etfs_unstacked, on=['sym','etf','date'], how='left').merge(var_etfs_unstacked, on=['etf','date'], how='left')
    result['etf_beta'] = (result['etf_cov'] / result['etf_var']).clip(PARAMS.BETA_LIMITS[0],PARAMS.BETA_LIMITS[1])
    result.drop(columns=['etf_cov','etf_var'],inplace=True)
    return result


class SymReturns(DataGrid):
    _SCHEMA = [
        ColumnDef('sym',CT.SYMBOL, isKey=True),
        ColumnDef('date', CT.DATE, isKey=True),
        ColumnDef('beta', CT.F32),  # market beta - same as in sym_stats, here for convienience
        ColumnDef('etf', CT.SYMBOL),  # most correlated etf - same as in sym_stats, here for convienience
        ColumnDef('etf_beta', CT.F32),  # most correlated etf's beta - same as in sym_stats, here for convienience
        ColumnDef('c2c', CT.F32), # close to close return from prev close to this date's close
        ColumnDef('o2c', CT.F32), # open to close return from this date's open to this date's close
        ColumnDef('c2cdn', CT.F32),  # close to close return from prev close to this date's close dollar neutral
        ColumnDef('o2cdn', CT.F32),  # open to close return from this date's open to this date's close dollar neutral
        ColumnDef('c2cbn', CT.F32),  # close to close return from prev close to this date's close beta neutral
        ColumnDef('o2cbn', CT.F32),  # open to close return from this date's open to this date's close beta neutral
        ColumnDef('c2csn', CT.F32),  # close to close return from prev close to this date's close sector neutral (beta ETF hedged)
        ColumnDef('o2csn', CT.F32),  # open to close return from this date's open to this date's close sector neutral (beta ETF hedged)
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.date.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('sym_returns',self._SCHEMA)


    def calcAndStoreSymReturns(self,startYr:int, endYr:int, KDB_ROOT: str = 'C://KDB_MARKET_DATA2/'):
        with get_md_conn() as q:
            self.kdbInitConnection(q)
        allYears = list(range(startYr, endYr + 1))
        batches = [allYears[offs: offs + 5] for offs in range(0, len(allYears), 5)]
        for index, years in enumerate(batches):
            logging.info(f'Processing yearly batch {index}/{len(batches)} for years {years[0]}-{years[-1]}')
            logging.info('Calculaing returns')
            returns = self.calcReturns(years[0], years[-1])
            returns [ 'year'] = returns.date.dt.year
            returns [ 'asof_date'] = np.datetime64(date.today())

            for year in years:
                yearData = returns [ returns.year == year]
                with get_md_conn() as q:
                    self.kdbInitPartitionTable(year,q)
                    self.upsertToKDB(q,KDB_ROOT,yearData)
                    self.saveKdbTableToDisk(q,year,KDB_ROOT)





    def calcReturns(self, startYr:int, endYr:int):
        '''
            Calculate varios returns for symbols ( c2c, o2c, c2cdn, o2cdn, c2cbn, o2cbn, c2csn, o2csn)
        '''
        universe = sorted(get_all_syms(date(startYr, 1, 1), date(endYr, 12, 31)))
        BATCH_SIZE=3000
        #symbols = sorted(list(set(universe[0:2000]+SYMSTAT_PARAMS.FACTOR_ETFS)))
        batches = [universe[offs: offs + BATCH_SIZE] for offs in range(0, len(universe), BATCH_SIZE)]
        all_returns = []
        for index, symbols in enumerate(batches):
            logging.info(f'Processing batch {index}/{len(batches)} for years {startYr}-{endYr}')
            symsWithETFS = sorted(list(set(symbols + PARAMS.FACTOR_ETFS)))
            query = '''
            {
            [syms;sd;ed]                     
            .md:select sym,date,adjusted_close,open,close from eodhd_price where year within (`year$sd;`year$ed), date>=sd,date<=ed,sym in `$syms,volume>0;
            //adding 1 to date below because stats as of T only become available R+1
            .symstats:select sym,date+1,beta,etf,etf_beta from sym_stats where year within (`year$sd-1;`year$ed),sym in `$syms;
            aj[`sym`date;.md;.symstats]
            }
            '''
            with get_md_conn() as q:
                md = self._sendSync(q, query, symsWithETFS, _kdbdt(date(startYr, 1, 1)-timedelta(days=10)), _kdbdt(date(endYr, 12, 31)))
            md['sym'] = md.sym.str.decode('utf-8')
            md['etf'] = md.etf.str.decode('utf-8')
            md.sort_values(['sym', 'date'], inplace=True)
            md.loc[ md.sym == md.shift(1).sym, 'c2c'] = (md.adjusted_close - md.shift(1).adjusted_close) / md.shift(1).adjusted_close
            md['o2c'] = (md.close - md.open) / md.open

            ##Add columns for most correlated etf beta and etf returns
            md = md.merge(md[['sym', 'date', 'c2c', 'o2c']], left_on=['etf', 'date'], right_on=['sym', 'date'],
                              suffixes=(None, '_etf'),how='left').drop(columns=['sym_etf'])
            ##Add columns for market returns
            MARKET = 'SPY'
            md = md.merge(md.loc[md.sym==MARKET,['date', 'c2c', 'o2c']],on='date',suffixes=('', '_market'),how='left')

            ##Add columns for dollar neutral returns
            md['c2cdn'] = md.c2c - md.c2c_market
            md['o2cdn'] = md.o2c - md.o2c_market
            ##Add columns for beta neutral returns
            md['c2cbn'] = md.c2c - md.beta * md.c2c_market
            md['o2cbn'] = md.o2c - md.beta * md.o2c_market

            ##Add columns for sector neutral returns
            md['c2csn'] = md.c2c - md.etf_beta * md.c2c_etf
            md['o2csn'] = md.o2c - md.etf_beta * md.o2c_etf

            all_returns.append(md)

        all_returns = pd.concat(all_returns)
        return all_returns[(all_returns.date.dt.year >= startYr) & (all_returns.date.dt.year <= endYr)]

    def getReturns(self,startDate=None,endDate=None,syms=None,columns=[]) ->pd.DataFrame:
        if startDate is None and endDate is None and syms is None:
            raise Exception("Must specify a where condition")
        startDate,endDate = startDate or date.min,endDate or date.max
        whereClause = f'where year >= `year$sd, year <=`year$ed,date>=sd,date<=ed'
        if syms is not None:
            whereClause += ',sym in `$syms'
        query = f'''
            {{[syms;sd;ed]
                (select {self.getColumnsWithCasts(columns)} from {self.name_} {whereClause})
                lj 2! select date,sym,adjusted_close,close,`float$volume from eodhd_price {whereClause}
            }}
        '''
        with get_md_conn() as q:
            data =  self._sendSync(q,query,syms or ['ALL'], _kdbdt(startDate),_kdbdt(endDate))
            return self.castToPython(data)

    def enrichWithReturns(self, trades, columns=[]) ->pd.DataFrame:
        ''' Enrich trades with returns
            :param trades: trades dataframe wich sym and date columns
        '''
        assert 'date' in trades.columns
        assert 'sym' in trades.columns
        temp = trades[['date','sym']].copy()
        temp['date']=temp.date.apply(_kdbdt)
        temp.meta = self.getqpythonMetaData()
        qcode = f'''
        {{
        [trades]
        trades: select sym,`date$date from trades;
        .temp:select {self.getColumnsWithCasts(columns)} from {self.name_} where sym in (exec sym from trades), date in (exec date from trades);
        trades lj 2!.temp
        }}
        '''
        with get_md_conn() as q:
            data = self._sendSync(q, qcode, temp)
        data = self.castToPython(data)
        return pd.merge(trades, data, on=['sym', 'date'], how='left')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
    b = SymReturns()
    #b.calcAndStoreSymReturns(2000,2023)
    #d = b.getReturns(date(2003, 1, 1),date(2004, 1, 1), syms=['GS', 'IBM'])
    trades = pd.DataFrame({'sym': ["IBM", "GS"], 'date': [date(2022, 7, 13), date(2022, 8, 13)]})
    data = b.enrichWithReturns(trades, [] )
    print(data)