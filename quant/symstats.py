import pandas as pd
from market_data.market_data import get_all_syms,get_md_conn,_kdbdt
import logging
from utils.datagrid import CT, ColumnDef, DataGrid
import numpy as np
from datetime import date
from utils.utils import get_all_trading_month_ends
from quant.symstat_params import SYMSTAT_PARAMS as PARAMS

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


class SymStats(DataGrid):
    _SCHEMA = [
        ColumnDef('sym',CT.SYMBOL, isKey=True),
        ColumnDef('date', CT.DATE, isKey=True),
        ColumnDef('beta', CT.F32), # Beta to market ( SPY )
        ColumnDef('etf', CT.SYMBOL), #Most correlated ETF
        ColumnDef('etf_corr',CT.F32),  # Most correlated ETF's correlation
        ColumnDef('etf_beta', CT.F32),  # Beta to most correlated ETF
        ColumnDef('volatility', CT.F32),  # annual volatility
        ColumnDef('ADV', CT.F32),  # AVERAGE DAILY VOLUME (21 day median) in shares
        ColumnDef('ADVD', CT.F32),  # AVERAGE DAILY VOLUME (21 day median) in dollars
        ColumnDef('year', CT.LONG, transformer = lambda frame: frame.date.dt.year, isPartition=True),
        ColumnDef('asof_date', CT.DATE, transformer=lambda frame: [np.datetime64(date.today())]*len(frame)),
    ]
    def __init__(self):
        super().__init__('sym_stats',self._SCHEMA)


    def calcAndStoreSymStats(self,startYr:int, endYr:int, KDB_ROOT: str = 'C://KDB_MARKET_DATA2/'):
        with get_md_conn() as q:
            self.kdbInitConnection(q)
        allYears = list(range(startYr, endYr + 1))
        batches = [allYears[offs: offs + 5] for offs in range(0, len(allYears), 5)]
        for index, years in enumerate(batches):
            logging.info(f'Processing yearly batch {index}/{len(batches)} for years {years[0]}-{years[-1]}')
            logging.info('Calculaing betas')
            betas = self.calcBetas(years[0], years[-1])
            logging.info('Calculating ADV and vol')
            advVol = self.calcADVandVol(years[0], years[-1])
            data = betas.merge(advVol, on=['sym','date'], how='outer')
            data [ 'year'] = data.date.dt.year
            data [ 'asof_date'] = np.datetime64(date.today())

            for year in years:
                yearData = data [ data.year == year]
                with get_md_conn() as q:
                    self.kdbInitPartitionTable(year,q)
                    self.upsertToKDB(q,KDB_ROOT,yearData)
                    self.saveKdbTableToDisk(q,year,KDB_ROOT)





    def calcADVandVol(self,startYr:int,endYr:int):
        month_ends = get_all_trading_month_ends()
        month_ends = [month_ends[(k, v)] for k, v in month_ends.keys() if k >= startYr and k <= endYr]
        universe = sorted(get_all_syms(date(startYr, 1, 1), date(endYr, 12, 31)))
        BATCH_SIZE=3000
        batches = [universe[offs: offs + BATCH_SIZE] for offs in range(0, len(universe), BATCH_SIZE)]
        all_data = []
        for index, symbols in enumerate(batches):
            logging.info(f'Processing ADV/VOL batch {index}/{len(batches)} for years {startYr}-{endYr}')
            query = '''
            { [ sd; ed; sampleDates; syms]
                .temp: update volatility: 252 mdev c2c , ADVD:med each {[w;s] {(neg &[x;count t])#t:y,z}[w]\[s]}[21;notional], ADV:med each {[w;s] {(neg &[x;count t])#t:y,z}[w]\[s]}[21;volume] 
                by sym from 
                update c2c:{(first[x]%':x)-1} adjusted_close by sym from 
                `date xasc select sym,date,volume,adjusted_close,notional:volume*close from eodhd_price where date>= sd, date <= ed, sym in `$syms;
                select sym,date,volatility,ADV,ADVD from .temp where date in sampleDates
            }
            '''

            sd = _kdbdt(date(startYr-1, 1, 1))
            ed = _kdbdt(date(endYr, 12, 31))
            with get_md_conn() as q:
                data = self._sendSync(q, query, sd,ed,[_kdbdt(d) for d in month_ends],symbols)
            data['sym'] = data.sym.str.decode('utf-8')
            data.sort_values(['sym', 'date'], inplace=True)
            all_data.append(data)
        return pd.concat(all_data)
    def calcBetas(self, startYr:int, endYr:int):
        '''
            Calculate betas for all symbols in the universe
        :param startYr:
        :param endYr:
        :return:
        '''
        month_ends = get_all_trading_month_ends()
        month_ends = [month_ends[(k, v)] for k, v in month_ends.keys() if k >= startYr-3 and k <= endYr]
        universe = sorted(get_all_syms(date(startYr, 1, 1), date(endYr, 12, 31)))
        BATCH_SIZE=3000
        #symbols = sorted(list(set(universe[0:2000]+SYMSTAT_PARAMS.FACTOR_ETFS)))
        batches = [universe[offs: offs + BATCH_SIZE] for offs in range(0, len(universe), BATCH_SIZE)]
        all_betas = []
        for index, symbols in enumerate(batches):
            logging.info(f'Processing batch {index}/{len(batches)} for years {startYr}-{endYr}')
            symsWithETFS = sorted(list(set(symbols + PARAMS.FACTOR_ETFS)))

            query = f'{{[dates;syms] select date,sym,adjusted_close from eodhd_price where date in dates ,volume>0, sym in `$syms}}'
            with get_md_conn() as q:
                md = self._sendSync(q, query, [_kdbdt(d) for d in month_ends],symsWithETFS)
            md['sym'] = md.sym.str.decode('utf-8')
            md.sort_values(['sym', 'date'], inplace=True)
            mrets = md.pivot(index='date', columns='sym', values='adjusted_close').pct_change() #indexed by month-end, columns are stocks
            #mrets = mrets[(mrets.index.year >= startYr) & (mrets.index.year <= endYr)] #monthly returns
            betas = calculate_rolling_betas(mrets)
            betas = betas[(betas.date.dt.year >= startYr) & (betas.date.dt.year <= endYr) & (betas.sym.isin(symbols))]
            all_betas.append(betas)
        all_betas = pd.concat(all_betas)
        return all_betas

    def enrichWithSymStats(self, trades, columns=[]) ->pd.DataFrame:
        ''' Enrich trades with symbol level statitics
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
        //0N!(.z.T,`$"SELECTING INTO TEMP");
        .temp:2!select {self.getColumnsWithCasts(columns)} from {self.name_} where 
        sym in (exec sym from trades), date<= (exec max date from trades), date >= ((exec min date from trades) - 60);
        //0N!(.z.T,`$"DONE SELECTING INTO TEMP. DOING AJ");
        .res:aj[`sym`date;trades;update `g#sym from .temp];
        //0N!(.z.T,`$"DONE");
        .res
        }}
        '''
        with get_md_conn() as q:
            data = self._sendSync(q, qcode, temp)
        data = self.castToPython(data)
        return pd.merge(trades, data, on=['sym', 'date'], how='left')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
    ss = SymStats()
    #ss.calcAndStoreSymStats(2000,2023)
    trades = pd.DataFrame({'sym': ["IBM", "GS"], 'date': [date(2022, 7, 13), date(2022, 8, 13)]})
    data = ss.enrichWithSymStats(trades, [] )
    print(data)
