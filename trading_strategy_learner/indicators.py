# name: Yu Bai; ID: ybai67


import os
import pandas as pd
import numpy as np
import datetime as dt
import time
from util import get_data, plot_data
from matplotlib.ticker import AutoMinorLocator


def cal_indicators(price, lookback=20):
    price_spy=pd.DataFrame(price.ix[0:,"SPY"])
    price_spy.columns = ["SPY"]
    price=price.iloc[0:,1:]

    # SMA
    sma=price.rolling(window=lookback,min_periods=lookback).mean()
    #p2sma = price/sma

    # BBP
    rolling_std=price.rolling(window=lookback,min_periods=lookback).std()
    top_band = sma + 2 * rolling_std
    bottom_band = sma - 2* rolling_std
    bbp = (price - bottom_band)/(top_band - bottom_band)

    # RSI
    daily_rets = cal_dr(price)
    rsi = cal_RSI(price, daily_rets,lookback)
    #rsi_spy=cal_RSI(price_spy,cal_dr(price_spy), lookback)

    # momentum
    momentum = price.copy()
    lookback2=lookback*2
    momentum.values[lookback2:, :] = momentum.values[lookback2:, :] / momentum.values[:-lookback2:, :] - 1
    momentum.values[:lookback2,:]=np.nan

    # moving std = rolling_std
    rel_std=rolling_std/sma

    return sma, bbp, rsi,momentum,rel_std, top_band, bottom_band, daily_rets


def cal_indicators4strategyLearner(sd, prices, lookback=20, standarize=False, verbose=False):
    sma, bbp, rsi, momentum, rel_std = cal_indicators(prices, lookback=lookback)[0:5]
    prices = prices.iloc[0:, 1:]
    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[prices >= sma] = 1  # days when price is higher
    sma_cross[1:] = sma_cross.diff()
    sma_cross.iloc[0] = 0
    debug = sma_cross.ix[(np.isfinite(sma.ix[:, prices.columns[0]].values))]
    firstNonNan = (debug.index)[0]
    sma_cross.ix[firstNonNan] = 0

    indicators = pd.concat([sma, bbp, rsi, momentum, rel_std, sma_cross],
                           keys=['p2sma', 'bbp', 'rsi', 'momentum', 'std', 'sma_cross'], axis=1)
    indicators.columns = ['p2sma', 'bbp', 'rsi', 'momentum', 'std', 'sma_cross']
    prices = prices.loc[prices.index >= sd]
    indicators = indicators.loc[indicators.index >= sd]
    indicators.ix[:,"p2sma"] = prices.ix[:,0] / indicators.ix[:,"p2sma"].values
    if standarize:
        indicators.ix[:, ['p2sma', 'bbp', 'rsi', 'momentum', 'std']] = stanadrizeIndicator(indicators.ix[:, ['p2sma', 'bbp', 'rsi', 'momentum', 'std']]).values
    if verbose: print indicators
    return prices, indicators


def cal_dr(price):
    daily_rets = price.copy()
    daily_rets.values[1:, :] = (price.values[1:, :] - price.values[:-1, :])
    daily_rets.iloc[0, :] = np.nan
    return daily_rets


def cal_RSI(price,daily_rets, lookback):
    up_rets = daily_rets[daily_rets>=0].fillna(0).cumsum(axis=0)
    down_rets = -1*daily_rets[daily_rets<0].fillna(0).cumsum(axis=0)
    up_gain = price.copy()
    up_gain.ix[:,:]=0
    up_gain.values[lookback:,:]=up_rets.values[lookback:,:]-up_rets.values[:-lookback,:]

    down_loss = price.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]
    # or use diff(period=lookback,axis=0)
    rs = (up_gain/lookback)/(down_loss/lookback)
    rsi=100-(100/(1+rs))
    rsi.ix[:lookback,:]=np.nan
    rsi[rsi==np.inf]=100
    return rsi

def getPrice(syms=["JPM"],sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31), includeSPY=False, extendPrior=None):
    if extendPrior is not None:
        extendedsd = sd - pd.to_timedelta(extendPrior*4, unit='d')
    else:
        extendedsd=sd
    dates = pd.date_range(extendedsd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    if extendPrior is not None:
        whichnewstart = np.arange(prices_all.shape[0])[prices_all.index >=sd][0]
        prices_all = prices_all.ix[whichnewstart-extendPrior*2:,:]
    prices_all = prices_all.fillna(method="ffill")
    prices_all = prices_all.fillna(method="bfill")
    if(includeSPY):
        return prices_all
    else:
        return prices_all[syms]  # only portfolio symbols

def stanadrizeIndicator(indicator):
        means=indicator.mean(axis=0)
        stds=indicator.std(axis=0)
        newindicator=(indicator-means)/stds
        return newindicator


def plot_SMA(price, sma, lookback=20):
    scaling=1.0/price.iloc[0, :]
    price = price / price.iloc[0, :]
    sma = sma * scaling #price.iloc[0:lookback,:].mean(axis=0)
    df = pd.concat([price, sma], keys=['Price', 'SMA'], axis=1)
    df.columns = ['Price', 'SMA']
    plot_df(df,title="Figure 1:   SMA (symbol="+price.columns[0]+")",
            color=["black","red"],style=["-","-"],
            figName="plot_Indicator_SMA.png")


def plot_RSI(price, rsi): #just 1 plot
    #normalize helper df price, scale rsi by 100 to 0-1
    scaling = 1.0 / price.iloc[0, :]
    price = price * scaling
    df = pd.concat([price, rsi],
                   keys=['Price', "RSI"], axis=1)
    df.columns = ['Price', 'RSI']
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df["Price"], 'black')
    ax1.set_title("Figure 3:   RSI (symbol=" + price.columns[0] + ")")
    ax1.set_xlim([df.index[0], df.index[-1]])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.5, 0.99))
    #ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["RSI"], 'r:')
    ax2.set_ylim(bottom=20,top=95)
    ax2.set_ylabel('RSI',color="red")
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 0.92))
    ax2.tick_params('y', colors='r')

    fig.autofmt_xdate()
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.tight_layout()
    fig.savefig("plot_Indicator_RSI.png")


def plot_std(price, sma, rolling_std):  # two subplots
    scaling=1.0/price.iloc[0, :]
    price = price / price.iloc[0, :]
    sma = sma * scaling
    df = pd.concat([price, sma, rolling_std], keys=['Price', 'SMA', 'Moving_STD'], axis=1)
    df.columns = ['Price', 'SMA', 'Moving_STD']
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df["Price"], 'black',df.index, df["SMA"], 'b')
    ax1.set_title("Figure 5:   (relative) Moving Standard Deviation")
    ax1.set_xlim([df.index[0], df.index[-1]])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='center left', bbox_to_anchor=(0.8, 0.6),labels=["Price","SMA"])
    ax3 = ax1.twinx()
    ax3.plot(df.index, df["Moving_STD"], 'r:')
    ax3.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    ax3.set_ylabel('Moving_STD',color="r")
    ax3.tick_params('y', colors='r')
    fig.autofmt_xdate()
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    fig.tight_layout()
    fig.savefig("plot_Indicator_MovingSTD.png")


def plot_df(df,title,color,style, figName, vline=None,logy=False):
    import matplotlib.dates as mdates
    ax = df.plot(title=title, figsize=(10, 5), fontsize=10, style=style,color=color,logy=logy)
    ax.set_xlabel('Date')
    ax.set_ylabel("Value")
    if(vline is not None):
        for each in vline.index:
            ax.axvline(x=each,color=vline.ix[each],linestyle="-",linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figName)

def author(self):
        return 'ybai67'


def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['JPM']
    start_val = 100000

    price = getPrice(syms=symbols, sd = start_date, ed = end_date,includeSPY=True, extendPrior=20)
    sma, bbp, rsi, momentum, rel_std, top_band, bottom_band, daily_rets = cal_indicators(price, lookback=20)
    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    df = pd.concat([price[symbols], sma, bbp, rsi, momentum, rel_std, top_band, bottom_band, daily_rets],
                   keys=['price', 'sma', 'bbp', 'rsi', 'momentum','std', 'top_band', 'bottom_band', 'daily_rets'], axis=1)
    df.columns=['price', 'sma', 'bbp', 'rsi','momentum','std', 'top_band', 'bottom_band', 'daily_rets']
    #df.to_csv("JPM_train.cvs")
    plot_SMA(price[symbols], sma, lookback=20)
    plot_RSI(price[symbols], rsi)
    plot_std(price[symbols],sma,rel_std)


if __name__ == "__main__":
    test_code()
