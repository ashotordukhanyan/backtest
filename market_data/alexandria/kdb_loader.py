import logging

from market_data.alexandria.alexandria import AlexandriaNewsSentiment, AlexandriaNewsDailySummary
from utils.qpthack import qconnection

if __name__ == '__main__':
    logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s')
    LOAD_NEWS_SENTIMENT = False
    LOAD_NEWS_DAILY_SUMMARY = True
    with qconnection.QConnection('localhost', 12345, pandas=True) as q:
        KDB_ROOT = "c:/KDB_MARKET_DATA2/"
        if LOAD_NEWS_SENTIMENT:
            logging.info('Loading news to KDB')
            DIR = 'c://Users/orduk/PycharmProjects/backtest/data/ALEXANDRIA/'
            FNAME = 'alexandria.sentiment.equity.DN.US.20000101-20240131.csv.gz'
            ANS = AlexandriaNewsSentiment()
            ANS.load_df_to_kdb(DIR+FNAME,q,KDB_ROOT)
        if LOAD_NEWS_DAILY_SUMMARY:
            logging.info('Loading news daily summary to KDB')
            ANDS = AlexandriaNewsDailySummary()
            ANDS.load_news_summary(q, KDB_ROOT,list(range(2000,2024)))
