import logging

from dateutil.relativedelta import relativedelta

from strategy.base import TraderParams, Trader
from utils.qpthack import qconnection, qtemporal, MetaData
import pandas as pd
from quant.symreturns import SymReturns
from quant.symstats import SymStats
from market_data.alexandria.alexandria import AlexandriaNewsDailySummary
from market_data.market_data import get_md_conn
from strategy import ARIMASignal, AvelanedaSignal
from datetime import date, timedelta

import pandas as pd
from utils.utils import cached_df, get_next_trading_days, get_last_trading_month_end
from symbology.compositions import get_composition_history
from typing import Tuple,List

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.base import  BaseEstimator
import numpy as np
import xgboost as xgb
from dataclasses import dataclass,field



@dataclass
class EnsembleParams(TraderParams):
    ''' Parameters for the ensemble strategy '''
    avel_signal_cufoffs: Tuple[float,float] = (-1.5,1.5)
    arima_target: str = 'c2cbn'
    news_sentiment_cutoffs: Tuple[float,float] = (0.,0.3)
    regressor : str = 'XGB' # #Ridge or Lasso or Linear or XGB
    features : List[str] =  field(default_factory=lambda: [
        'prediction', 'predicted_se',  # from ARIMA ( also modelP, modelQ )?
        'signal', 'oualpha', 'ougamma', 'oubeta',  # from Avelaneda ?
        'Sentiment', 'Mentions', 'Relevance', 'Confidence',  # from Alexandria
        'ADVD', 'beta', 'volatility', 'etf',
        'has_arima_signal', 'has_avel_signal', 'has_news_signal'
    ])

DEFAULT_ENSEMBLE_PARAMS = EnsembleParams(universe='IWV', target='o2cbn')

class ReturnsEstimator:
    def __init__(self,estimator:BaseEstimator,r2_score:float,prediction_quantiles:List[float]):
        self.estimator_ = estimator
        self.r2_score_ = r2_score
        self.prediction_quantiles_ = prediction_quantiles
class EnsembleTrader(Trader):
    def __init__(self,params:EnsembleParams = DEFAULT_ENSEMBLE_PARAMS):
        super().__init__(params)

    def _preprocess(self,signals:pd.DataFrame)->pd.DataFrame:
        signals = signals[~signals[self.params_.target].isna()].copy()
        signals['has_arima_signal'] = signals['prediction'].notnull().astype(int)
        signals['has_avel_signal'] = signals['signal'].notnull().astype(int)
        signals['has_news_signal'] = signals['Mentions'].notnull().astype(int)
        return signals
    def fit(self,signals:pd.DataFrame) -> BaseEstimator:
        columnTransformations = []
        numerical_features = [ f for f in self.params_.features if f not in ['etf','has_arima_signal','has_avel_signal','has_news_signal']]
        if numerical_features:
            rescale = make_pipeline(SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=0),StandardScaler())
            columnTransformations.append(('rescale', rescale, numerical_features) )

        categorical_features  = [ f for f in self.params_.features if f in [ 'etf' ]]
        if categorical_features:
            etfOneHot = make_pipeline(SimpleImputer(missing_values='', strategy='constant', fill_value='SPY'),
                                      OneHotEncoder())
            columnTransformations.append(('etfs', etfOneHot, categorical_features))

        transformer = ColumnTransformer(columnTransformations)

        if self.params_.regressor == 'Linear':
            regressor = LinearRegression()
        elif self.params_.regressor== 'Ridge':
            regressor = Ridge()
        elif self.params_.regressor == 'Lasso':
            regressor = Lasso()
        elif self.params_.regressor == 'XGB':
            regressor = xgb.XGBRegressor()
        else :
            raise Exception(f"Unknown regressor {self.params_.regressor}")

        pipeline = make_pipeline(transformer,regressor)
        model=TransformedTargetRegressor(regressor=pipeline,transformer=StandardScaler())
        ytrue = signals[self.params_.target]
        model.fit(signals[self.params_.features],ytrue)
        ypred = model.predict(signals[self.params_.features])
        r2 = r2_score(ytrue,ypred)
        #logging.info(f'Fitted model with R2={r2}')
        NUM_BUCKETS = 5
        quantiles = [np.quantile(ypred,x) for x in np.arange(0,1,1/NUM_BUCKETS)]
        return ReturnsEstimator(model,r2,quantiles)
    def score(self,signals:pd.DataFrame,model:ReturnsEstimator)->float:
        ypred = model.estimator_.predict(signals[self.params_.features])
        predQuantiles = np.digitize(ypred, model.prediction_quantiles_)
        yActual = signals[self.params_.target]
        def robustMean(data,m=3):
            return data[abs(data - np.mean(data)) < m * np.std(data)].mean()

        yActualQuantiles = [robustMean(yActual[predQuantiles==i]) for i in np.unique(predQuantiles)]
        r2Score = r2_score(yActual, ypred)
        return r2Score,yActualQuantiles
    def runTraining(self,startDate:date = date(2007,7,1),endDate:date = date(2023,12,1)):
        simDate = startDate
        while (simDate < endDate):
            TRAIN_PERIOD_START = simDate - self.params_.trainPeriod
            TRAIN_PERIOD_END = simDate - relativedelta(days=1)
            TEST_PERIOD_START = simDate
            TEST_PERIOD_END = simDate + self.params_.trainFrequency# - relativedelta(days=1)
            simDate = simDate + self.params_.trainFrequency
            logging.info(f'Training for {TRAIN_PERIOD_START} - {TRAIN_PERIOD_END} Testing for {TEST_PERIOD_START} - {TEST_PERIOD_END}')
            signals = getAllSignals(TRAIN_PERIOD_START,TRAIN_PERIOD_END,AVEL_SIGNAL_CUTOFFS=self.params_.avel_signal_cufoffs,ARIMA_TARGET=self.params_.arima_target,
                                    NEWS_SENTIMENT_CUTOFFS=self.params_.news_sentiment_cutoffs, NEWS_UNIVERSE=self.params_.universe)
            #remove outliers
            OUTLIER_RETURN_CUTOFF = 0.5
            signals = signals[(signals.c2c.abs()<=OUTLIER_RETURN_CUTOFF) & (signals.o2c.abs()<=OUTLIER_RETURN_CUTOFF)]
            model = self.fit(self._preprocess(signals))
            inSampleR2,inSampleQuantiles = self.score(self._preprocess(signals), model)
            logging.info(f'In Sample R2={inSampleR2:.4f} Bucket returns {["{:.4f}".format(x) for x in inSampleQuantiles]}')
            logging.info(f'In Sample R2={inSampleR2:.4f} Buckets        {["{:.4f}".format(x) for x in model.prediction_quantiles_]}')
            testSignals = getAllSignals(TEST_PERIOD_START,TEST_PERIOD_END,AVEL_SIGNAL_CUTOFFS=self.params_.avel_signal_cufoffs,ARIMA_TARGET=self.params_.arima_target,
                                        NEWS_SENTIMENT_CUTOFFS=self.params_.news_sentiment_cutoffs, NEWS_UNIVERSE=self.params_.universe)
            r2,quantiles = self.score(self._preprocess(testSignals),model)
            logging.info(f'Validation R2={r2:.4f} Bucket returns {["{:.4f}".format(x) for x in quantiles]}')
@cached_df
def getAllSignals(sd:date,ed:date,AVEL_SIGNAL_CUTOFFS = (-1.5,1.5),
                  ARIMA_TARGET='c2cbn',
                  NEWS_SENTIMENT_CUTOFFS=[0.,0.3],
                  NEWS_UNIVERSE='IWV')->pd.DataFrame:
    arimaS = ARIMASignal().retrieveSignal(sd,ed,target=ARIMA_TARGET,columns=\
        ['sym', 'date', 'target', 'prediction', 'predicted_se', 'model_order','modelP', 'modelQ'])
    avelS = AvelanedaSignal().retrieveSignal(sd,ed,columns=['sym', 'date', 'signal', 'oualpha', 'ougamma', 'oubeta'])
    avelS = avelS[(avelS.signal < AVEL_SIGNAL_CUTOFFS[0]) | (avelS.signal > AVEL_SIGNAL_CUTOFFS[1])]
    ANDS = AlexandriaNewsDailySummary()
    with get_md_conn() as q:
        newsS = ANDS.get_sentiments_by_effective_date(sd, ed, q, 'ALL')
    newsS.drop(columns=['MarketImpactScore', 'Prob_POS', 'Prob_NTR', 'Prob_NEG'], inplace=True)
    newsS.rename(columns={ 'Ticker': 'sym'}, inplace=True)
    ## move the date
    newsS['date'] = get_next_trading_days(newsS.EffectiveDate.tolist())
    ## Effective date represents the date of the news. We are interested in "next days" o2c returns -so set date to be next business date
    newsS = newsS[(newsS.Sentiment < NEWS_SENTIMENT_CUTOFFS[0]) | (newsS.Sentiment > NEWS_SENTIMENT_CUTOFFS[1])]

    if NEWS_UNIVERSE is not None:
        ## restrict news only to symbols that were in a given composition as of.
        ## compositions are stored for month-ends only
        compositions = get_composition_history(NEWS_UNIVERSE, sd-timedelta(days=45), ed)
        newsS['EffectiveMonthEnd'] = newsS['EffectiveDate'].apply(get_last_trading_month_end)
        newsS = pd.merge(newsS, compositions[['ticker', 'asof_date']], left_on=['sym', 'EffectiveMonthEnd'], right_on=['ticker', 'asof_date'], how='inner')

    signals = pd.merge(pd.merge(arimaS, avelS, on=['sym', 'date'], how='outer'),newsS,on=['sym','date'],how='outer')
    RETURN_MEASURES = ['c2c','o2c','c2cbn','o2cbn','c2cdn','o2cdn','c2csn','o2csn']
    signals = SymReturns().enrichWithReturns(signals,columns=['sym','date']+RETURN_MEASURES)
    signals = SymStats().enrichWithSymStats(signals, columns=['etf', 'ADVD', 'beta', 'volatility', 'sym', 'date'])
    return signals

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',filename=None)
    smallParams = EnsembleParams(universe='IWV', target='o2cbn', regressor='Linear', features = [ 'prediction',"signal","Sentiment" ])
    strategy = EnsembleTrader(params=smallParams)
    logging.info('Starting training')
    strategy.runTraining()
    logging.info('Training done')
