#implementing a BestPossibleStrategy object
#implement testPolicy() which returns a trades data frame df_trades:
# A data frame whose values represent trades for each day. Legal values are +1000.0 indicating a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
# Values of +2000 and -2000 for trades are also legal so long as net holdings are constrained to -1000, 0, and 1000.

#"main": call marketsimcode as necessary to generate the plots used in the report
#Commission: $0.00, Impact: 0.00
# implement benchmark calculation here so it can be called when needed

import os
import pandas as pd
from indicators import *
from marketsimcode import *

def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed= dt.datetime(2009,12,31), sv = 100000):
    prices=getPrice(symbol,sd,ed)
    df_trades = prices.copy()
    df_trades[:]=0
    forecast = prices.copy()
    #peak into future, if up long Max shares allowed, if down short today, if no change or will exceed allowance, hold
    forecast.values[0:-1, :] = (prices.values[1:, :] - prices.values[:-1, :])
    forecast.values[-1,0]=0  #no trade last day as no future data as direction
    forecast = np.sign(forecast)
    # eliminate the entry that violate the allowance
    for i in range(forecast.shape[0]):
        holdings = df_trades.values[:(i+1),:].sum()
        #print holdings
        df_trades.iloc[i, :] = 1000 * forecast.values[i, :] - holdings
    holdings = df_trades.cumsum(axis=0)
    return df_trades

def benchmarkPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed= dt.datetime(2009,12,31), sv = 100000):
    prices=getPrice(symbol,sd,ed)
    df_trades = prices.copy()
    df_trades[:]=0
    df_trades.iloc[0,:] = 1000
    return df_trades

def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['JPM']
    start_val = 100000
    impact= 0
    comission =0
    df_benchmark = benchmarkPolicy(symbols, start_date, end_date, start_val)
    df_trades = testPolicy(symbols, start_date, end_date, start_val)

    port_benchmark = compute_portvals(df_benchmark,start_val = 100000, commission=comission, impact=impact)
    port_best = compute_portvals(df_trades, start_val=100000, commission=0, impact=0)

    cv_benchmark, dr_benchmark, std_dr_benchmark, sharpe_benchmark=compute_portfolio_stats(port_benchmark.iloc[:,0])
    cv_best, dr_best, std_dr_best, sharpe_best=compute_portfolio_stats(port_best.iloc[:,0])

    print
    print "Symbols:", symbols
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Start Value:", start_val
    print "Impact:", impact
    print "Commission:", comission

    print "In-Sample Benchmark final portfolio =\t"+str(port_benchmark.iloc[-1,0])
    print "In-Sample BestStrategy final portfolio =\t"+str(port_best.iloc[-1,0])

    print "In-Sample Benchmark cv =\t"+str(cv_benchmark)
    print "In-Sample BestStrategy cv =\t"+str(cv_best)

    print "In-Sample Benchmark dr =\t"+str(dr_benchmark)
    print "In-Sample BestStrategy dr =\t"+str(dr_best)

    print "In-Sample Benchmark std_dr =\t"+str(std_dr_benchmark)
    print "In-Sample BestStrategy std_dr =\t"+str(std_dr_best)

    print "In-Sample Benchmark sharpe ratio =\t"+str(sharpe_benchmark)
    print "In-Sample BestStrategy sharpe ratio =\t"+str(sharpe_best)
    print

    df = pd.concat([port_benchmark, port_best],
                   keys=['Benchmark', "BestPossible"], axis=1)
    df.columns = ['Benchmark', "BestPossible"]
    df=df/df.iloc[0,:]
    plot_df(df,title="Figure 6:   In-Sample Portfolio of BestPossible Strategy",
            color=["blue", "black"],style=["-","-"],
            figName="Portfolio_BestPossible_InSample.png",vline=None)


if __name__ == "__main__":
    test_code()
