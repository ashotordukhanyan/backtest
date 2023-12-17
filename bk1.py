from zipline.api import order_target, record, symbol,attach_pipeline,schedule_function,pipeline_output
from zipline import run_algorithm
from datetime import date,datetime
import pandas as pd
import matplotlib.pyplot as plt
from zipline.pipeline.factors.basic import DailyReturns
from zipline.pipeline import CustomFactor,Pipeline
from zipline.utils.events import date_rules
from zipline.pipeline.factors import RSI

def make_pipeline():
    dr = DailyReturns()

    return Pipeline(
       columns={
           "daily_returns": dr,
       },
    )

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')
    # running at the start of the day each day.
    attach_pipeline(make_pipeline(),'DailyReturnsCalc')
    #schedule_function(doStuff, date_rules.every_day())

def before_trading_start(context, data):
    context.daily_returns = pipeline_output("DailyReturnsCalc")

def doStuff(context,data):
    if hasattr(context,'pipeline_data'):
        pipeline_data = context.pipeline_data
        print(f'Got pipeline data of shape {pipeline_data.shape}')
    else:
        print('no pipeline data')

def handle_data(context, data):
    # Skip first 300 days to get full windows
    #print(f'hd context.i {type(data)} {data.current_dt}')
    if context.i % 100 == 0:
        print(f"{context.i} {data.current(context.asset, 'price')} {data.current(symbol('GS'),'price')}")
    context.i += 1
    if context.i < 300:
        return
    daily_returns = context.daily_returns
    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()
    if context.i%100 == 0:
        print(data.history(context.asset, 'price', bar_count=100, frequency="1d"))
    # Trading logic
    if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)

def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(212)
    perf['AAPL'].plot(ax=ax2)
    perf[['short_mavg', 'long_mavg']].plot(ax=ax2)

    perf_trans = perf.loc[[t != [] for t in perf.transactions]]
    buys = perf_trans.loc[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.loc[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax2.plot(buys.index, perf.short_mavg.loc[buys.index],
             '^', markersize=10, color='m')
    ax2.plot(sells.index, perf.short_mavg.loc[sells.index],
             'v', markersize=10, color='k')
    ax2.set_ylabel('price in $')
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    result = run_algorithm(start=pd.Timestamp(2012,1,1),end=pd.Timestamp(2018,1,1),initialize=initialize,
                  handle_data=handle_data,capital_base=1e6,bundle='quandl', analyze=analyze,
                           before_trading_start=before_trading_start)
    print('done')