from datetime import date
import logging
from symbology.compositions import get_composition_history
from tqdm import tqdm

logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(message)s')


r1 = get_composition_history('IWB',date(2000,1,1),date.today())
r2 = get_composition_history('IWM',date(2000,1,1),date.today())
r3 = get_composition_history('IWV',date(2000,1,1),date.today())
sp500 = get_composition_history('IVV',date(2000,1,1),date.today())

tickers = set()
for c in [r1,r2,r3,sp500]:
    tickers = tickers.union(set(c.ticker.unique()))

asofdates = sorted(r1.asof_date.unique().tolist()) ## same for all

for index in range(len(asofdates)-1):
    d1 = asofdates[index]
    d2 = asofdates[index+1]

    c1 = set(r3[r3.asof_date == d1]['ticker'].to_list())
    c2 = set(r3[r3.asof_date == d2]['ticker'].to_list())

    adds,deletes = c2-c1,c1-c2
    if adds or deletes:
        print(f'from {d1.strftime("%Y%m%d")} to {d2.strftime("%Y%m%d")} {len(adds)} adds and {len(deletes)} deletes')
        if d2.year == 2023:
            print(f' ADDS {" ".join(adds)} deletes {" ".join(deletes)}')




