#marketsim code that accepts a "trades" data frame (instead of a file)

import os
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders, start_val = 100000, commission=9.95, impact=0.005):
    # orders is Date x symbol df with 0, +/- shares per day
    orders.sort_index(inplace=True)

    #call get_data to get adj close prices of syms and within start/end dates
    dates = pd.date_range((orders.index).format()[0],(orders.index).format()[-1])
    prices = get_data(np.unique(orders.columns).tolist(), dates=dates)
    prices = prices.iloc[:,1:] #remove SPY

    # add cash col -> one copy for prices,
    prices["CASH"] = pd.Series(np.ones(prices.shape[0]),index=prices.index)

    signs= np.sign(orders.values)
    absSigns = np.abs(np.sign(orders.values))

    cash = -1.0*(orders * prices[orders.columns])*(1 + signs * impact) - commission * absSigns
    #print ['CASH'] +  orders.columns.values.tolist()
    trades = pd.concat([cash, orders], keys=['CASH'] +  orders.columns.values.tolist(), axis=1)
    trades.columns=['CASH'] +  orders.columns.values.tolist()
    debug =9
    # holdings (init with copy, fill with 0 & cash for 1st row, for each row in holdings=(itself for 1st row|pre row otherwise) + same row from trades
    holdings = trades.copy()
    holdings[:] = 0
    holdings = holdings + np.cumsum(trades.values, axis=0)
    holdings["CASH"] = holdings["CASH"] + start_val

    values = prices * holdings
    # values.sum(axis=1)
    portvals = pd.DataFrame(values.sum(axis=1),index=values.index)

    return portvals


def compute_portfolio_stats(port_val,rfr = 0.0, sf = 252.0):
    cumReturn = (port_val[-1]/port_val[0])-1
    periodReturn = port_val.copy()
    periodReturn[1:]=(periodReturn[1:]/periodReturn[:-1].values)-1
    periodReturn.iloc[0] = 0
    avgPeriodReturn = periodReturn[1:].mean()
    stdPeriodReturn = periodReturn[1:].std()
    sharpe = np.sqrt(sf) * (avgPeriodReturn - rfr) / stdPeriodReturn
    return cumReturn, avgPeriodReturn, stdPeriodReturn, sharpe

def test_code():
    of = pd.DataFrame([1000,0,-2000,0,0,0,1000,-1000,2000,-1000],
                      index=pd.date_range(pd.datetime(2008,1,7),pd.datetime(2008,1,18),freq="B"),
                      columns=["JPM"])
    sv = 100000

    # Process orders
    portvals = compute_portvals(orders=of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    print portvals
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = (portvals.index).format()[0]
    end_date = (portvals.index).format()[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
