"""indicators as functions that operate on dataframes.
 The "main" code in indicators.py should generate the charts that illustrate your indicators in the report
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
import time
from util import get_data, plot_data
from matplotlib.ticker import AutoMinorLocator


def cal_indicators(price, volume, lookback=20):
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

    #OBV
    #obv = volume.copy()
    #obv.iloc[0,:]=0
    #obv.iloc[1:,:]=volume.iloc[1:,:] * np.sign(daily_rets.iloc[1:,:])
    #obv = obv.cumsum(axis=0)

    return sma, bbp, rsi,momentum,rel_std, top_band, bottom_band, daily_rets


def cal_trend(data, lookback=40): #obsolete
    trend=data.copy()
    trend.values[lookback:, :] = (trend.values[lookback:, :] - trend.values[:-lookback:, :])
    trend.values[:lookback, :] = np.nan
    return trend

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

def getPrice(syms=["JPM"],sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31), includeSPY=False):
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = prices_all.fillna(method="ffill")
    prices_all = prices_all.fillna(method="bfill")
    if(includeSPY):
        return prices_all
    else:
        return prices_all[syms]  # only portfolio symbols

def getVolume(syms=["JPM"],sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31), includeSPY=False):
    dates = pd.date_range(sd, ed)
    vol_all = get_data(syms, dates,addSPY=True, colname = 'Volume')  # automatically adds SPY
    vol_all = vol_all.fillna(method="ffill")
    vol_all = vol_all.fillna(method="bfill")
    if(includeSPY):
        return vol_all
    else:
        return vol_all[syms]  # only portfolio symbols


def plot_SMA(price, sma, lookback=20):
    scaling=1.0/price.iloc[0, :]
    price = price / price.iloc[0, :]
    sma = sma * scaling #price.iloc[0:lookback,:].mean(axis=0)
    df = pd.concat([price, sma], keys=['Price', 'SMA'], axis=1)
    df.columns = ['Price', 'SMA']
    plot_df(df,title="Figure 1:   SMA (symbol="+price.columns[0]+")",
            color=["black","red"],style=["-","-"],
            figName="plot_Indicator_SMA.png")

def plot_BBP(price, sma, top_band, bottom_band, bbp): #
    scaling=1.0/price.iloc[0, :]
    price = price * scaling
    sma = sma * scaling
    top_band = top_band * scaling
    bottom_band = bottom_band * scaling
    df = pd.concat([price, top_band, bottom_band, bbp], keys=['Price', 'top Bollinger band', 'bottom Bollinger band', "%BB"], axis=1)
    df.columns = ['Price', 'top Bollinger band', 'bottom Bollinger band', "%BB"]

    plot_df(df,title="Figure 2:   %BB (symbol="+price.columns[0]+")",
            color=["black","dodgerblue","dodgerblue","red"],style=["-","-","-",":"],
            figName="plot_Indicator_BBP.png")


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

    '''
    df = pd.concat([price, rsi,rsi_spy],
                   keys=['Price', "RSI_JPM","RSI_SPY"], axis=1)
    df.columns = ['Price', 'RSI(JPM)',"RSI(SPY)"]
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df["Price"], 'black')
    ax1.set_title("Indicator: RSI")
    ax1.set_xlim([df.index[0], df.index[-1]])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.5, 0.99))
    #ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["RSI(SPY)"], 'b:')
    ax2.set_ylim(bottom=20,top=95)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 0.92))
    ax3 = ax1.twinx()
    ax3.plot(df.index, df["RSI(JPM)"], 'r:')
    ax3.set_ylim(bottom=20,top=95)
    ax3.legend(loc='upper left', bbox_to_anchor=(0.5, 0.85))
    fig.autofmt_xdate()
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.savefig("plot_Indicator_RSI.png")
    '''
#    plot_df(df,title="Indicator: RSI (symbol=" + price.columns[0] + ")",
#            color=["black", "red"],style=["-",":",":"],
#            figName="plot_Indicator_RSI.png")

def plot_momentum(price, sma,momentum): #1 or 2?
    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[sma>=price]=1 #days when price is higher
    sma_cross[1:]=sma_cross.diff()
    sma_cross.iloc[0]=0
    firstNonNan=(sma_cross.ix[(np.isfinite(sma["JPM"].values))].index)[0]
    sma_cross.ix[firstNonNan]=0

    scaling = 1.0 / price.iloc[0, :]
    price = price * scaling
    sma = sma*scaling

    sel=(sma_cross.ix[sma_cross["JPM"]!=0]).index.to_pydatetime()
#    pd.concat([momentum[sma_cross["JPM"]!=0],sma_cross[sma_cross["JPM"]!=0]],axis=1).to_csv("moment_20_atCross.cvs")
    df = pd.concat([price,sma, momentum],
                   keys=['Price', "SMA","Momentum"], axis=1)
    df.columns = ['Price',"SMA", 'Momentum']
    plot_df(df,title="Figure 4:   Momentum (symbol=" + price.columns[0] + ")",
                 color=["black","dodgerblue", "red"],style=["-","-",":"],
                 figName="plot_Indicator_Momentum.png",vline=None)
'''
def plot_obv(price,  obv):
    # normalize price, obv, obv-price (or obv/price?)
    scaling = 1.0 / price.iloc[0, :]
    priceNorm = (price-price.mean(axis=0))/price.std(axis=0)
    obvNorm = (obv-obv.mean(axis=0))/obv.std(axis=0)
    price_trend=cal_trend(priceNorm,40)
    obv_trend=cal_trend(obvNorm,40)
    df = pd.concat([price, obv, obv_trend-price_trend],
                   keys=['Price', "OBV","OBV.trend-Price.trend"], axis=1)
    df.columns = ['Price', "OBV","OBV.trend-Price.trend"]
    #plot_df(df,title="Indicator: OBV (symbol=" + price.columns[0] + ")", \
    #             color=["black","dodgerblue","red"],style=["-","-",":"],
    #            figName="plot_Indicator_OBV.png")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 9))
    df.plot(title="Indicator: OBV (symbol="+price.columns[0]+")",
            y=["Price"], ax=axes[0],color="black")
    df.plot(y=["OBV"], ax=axes[1],color="red",style=":")
    df.plot(y=["OBV.trend-Price.trend"], ax=axes[2],color="red",style=":")
    axes[2].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[1].set_ylabel("OBV")
    axes[2].set_ylabel("OBV.trend-Price.trend")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    fig.savefig("plot_Indicator_OBV.png")
'''

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
            ax.axvline(x=each,color=vline.ix[each],linestyle="-",linewidth=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figName)


def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['JPM']
    start_val = 100000

    price = getPrice(syms=symbols, sd = start_date, ed = end_date,includeSPY=True)
    volume= getVolume(syms=symbols, sd=start_date, ed=end_date)
    sma, bbp, rsi, momentum, rel_std, top_band, bottom_band, daily_rets = cal_indicators(price, volume,lookback=20)
    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    df = pd.concat([price[symbols], volume, sma, bbp, rsi, momentum, rel_std, top_band, bottom_band, daily_rets],
                   keys=['price', 'volume', 'sma', 'bbp', 'rsi', 'momentum','std', 'top_band', 'bottom_band', 'daily_rets'], axis=1)
    df.columns=['price', 'volume', 'sma', 'bbp', 'rsi','momentum','std', 'top_band', 'bottom_band', 'daily_rets']
    #df.to_csv("JPM_train.cvs")
    plot_SMA(price[symbols], sma, lookback=20)
    plot_BBP(price[symbols], sma, top_band, bottom_band,bbp)
    plot_RSI(price[symbols], rsi)
    plot_momentum(price[symbols],sma, momentum)
    plot_std(price[symbols],sma,rel_std)


if __name__ == "__main__":
    test_code()
