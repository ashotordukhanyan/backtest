from market_data.market_data import get_all_syms,get_md_conn, get_price_volume
import logging
import numpy as np
from utils.qpthack import qconnection,qtemporal,MetaData
import pandas as pd
from datetime import date,timedelta
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from symbology.sym_lifecycle import get_all_syms_active_on_period
from utils.utils import get_next_trading_day
from symbology.compositions import get_spx_constituents
logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')

SIM_START_DATE = date(2001,1,3)
SIM_END_DATE = date(2023,9,1)
TRAIN_PERIOD = relativedelta(months=12)

TRADE_DATE = SIM_START_DATE
while TRADE_DATE < SIM_END_DATE:
    TRAIN_START_DATE = TRADE_DATE - TRAIN_PERIOD
    TRAIN_END_DATE = TRADE_DATE - relativedelta(days=1)

    active_syms = set(get_all_syms_active_on_period(get_next_trading_day(TRAIN_START_DATE),TRAIN_END_DATE))
    spx_syms =set(get_spx_constituents(TRAIN_START_DATE)).intersection(set(get_spx_constituents(TRAIN_END_DATE)))
    bad= spx_syms - active_syms
    syms = list(spx_syms.intersection(active_syms))

    logging.warning(f'{TRADE_DATE.strftime("%Y%m%d")} Missing data for {len(bad)} syms: {",".join(sorted(bad))}')
    TRADE_DATE = TRADE_DATE + relativedelta(months=1)

#pv = get_price_volume(syms,TRAIN_START_DATE,TRAIN_END_DATE)
#pv.sort_values(['sym', 'date'], inplace=True)
#pv.loc[pv['sym'] == pv['sym'].shift(), 'log_return'] = \
#    np.log(pv['adjusted_close']) - np.log(pv['adjusted_close'].shift())


from sklearn.decomposition import PCA


# #shape data for PCA analysis
# pca_data = pv.pivot(index='date',columns='sym',values='log_return').fillna(0)
# pca_data = pca_data - pca_data.mean()
# pca_data = pca_data / pca_data.std()
# #find nans in pca_data
# nan_mask = np.isnan(pca_data)
# nan_mask = nan_mask.any(axis=1)
# pca_data = pca_data[~nan_mask]
# pca_data = pca_data.T
#
# pca = PCA(n_components=10)
# pca.fit(pca_data) #(samples/features)
