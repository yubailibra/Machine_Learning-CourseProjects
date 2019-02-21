# name: Yu Bai; ID: ybai67

#implement testPolicy() which returns a trades data frame according to the Manual Strategy

import os
import pandas as pd
from indicators import *
from marketsimcode import *

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        #prices = getPrice(symbol, sd, ed, includeSPY=True)
        #sma, bbp, rsi, momentum, rel_std = cal_indicators(prices, lookback=20)[0:5]

        prices = getPrice(syms=[symbol], sd=sd, ed=ed, includeSPY=True, extendPrior=20)
        sma, bbp, rsi, momentum, rel_std = cal_indicators(prices, lookback=20)[0:5]
        prices = prices.iloc[0:, 1:]
        prices = prices.loc[prices.index >= sd]
        sma = sma.loc[sma.index >= sd]
        bbp = bbp.loc[bbp.index >= sd]
        rsi = rsi.loc[rsi.index >= sd]
        momentum = momentum.loc[momentum.index >= sd]
        rel_std = rel_std.loc[rel_std.index >= sd]

        sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
        sma_cross[prices >= sma] = 1  # days when price is higher
        sma_cross[1:] = sma_cross.diff()
        sma_cross.iloc[0] = 0
        debug = sma_cross.ix[(np.isfinite(sma.ix[:, prices.columns[0]].values))]
        firstNonNan = (debug.index)[0]
        sma_cross.ix[firstNonNan] = 0

        positions = prices.copy()
        positions[:] = np.nan
        # sma (compare to price), bbp, rel_std, rsi, momentum
        positions[(prices / sma < 0.95) & (bbp < 0) & (rel_std <= 0.07) & (rsi < 35) & (momentum < -0.1)] = 1  #
        positions[(prices / sma > 1.05) & (bbp > 1) & (rel_std <= 0.07) & (rsi > 60) & (momentum > 0.12)] = -1  #
        positions[(sma_cross != 0) & (rel_std <= 0.06) & (momentum.abs() > 0.2)] = 0
        positions.ffill(inplace=True)
        positions.fillna(0, inplace=True)

        df_trades = positions.copy()
        df_trades[1:] = df_trades.diff() * 1000
        holdings = df_trades.cumsum(axis=0)
        return df_trades
