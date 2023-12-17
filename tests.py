from utils.qpthack import QConnection
import pandas as pd
from symbology.compositions import get_broad_etfs

etfs = get_broad_etfs()
Q = '{[syms] select sd:min date, ed: max date, days:count i by date from daily_prices where Ticker in '
