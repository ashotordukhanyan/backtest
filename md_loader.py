from datetime import date
import logging
from market_data.market_data import get_daily_market_data
from tqdm import tqdm

logging.basicConfig(filename=None,level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
md = get_daily_market_data('IWB IWM IWV IVV'.split(),date(2000,1,1),date(2023,9,1))
logging.info(f"Got back {md.shape}")

del(md)
