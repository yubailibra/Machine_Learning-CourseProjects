#implementing a ManualStrategy object
#implement testPolicy() which returns a trades data frame
#"main": call marketsimcode as necessary to generate the plots used in the report
#Commission: $9.95, Impact: 0.005

import os
import pandas as pd
from indicators import *
from marketsimcode import *
import BestPossibleStrategy

def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed= dt.datetime(2009,12,31), sv = 100000):
    prices = getPrice(symbol, sd, ed,includeSPY=True)
    volume= getVolume(symbol, sd, ed)
    sma, bbp, rsi, momentum, rel_std = cal_indicators(prices,volume,lookback=20)[0:5]

    prices=prices.iloc[0:,1:]
    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[prices>=sma]=1 #days when price is higher
    sma_cross[1:]=sma_cross.diff()
    sma_cross.iloc[0]=0
    debug=sma_cross.ix[(np.isfinite(sma.ix[:,symbol[0]].values))]
    firstNonNan=(debug.index)[0]
    sma_cross.ix[firstNonNan]=0

    positions = prices.copy()
    positions[:]=np.nan
    #sma (compare to price), bbp, rel_std, rsi, momentum
    positions[(prices/sma<0.95) & (bbp<0) & (rel_std<=0.07) & (rsi<35) & (momentum<-0.1)]=1 #
    positions[(prices/sma>1.05) & (bbp>1) & (rel_std<=0.07) & (rsi>60) & (momentum>0.12)]=-1 #
    positions[(sma_cross!=0) & (rel_std<=0.06) & (momentum.abs()>0.2)] =0
    positions.ffill(inplace=True)
    positions.fillna(0,inplace=True)

    df_trades=positions.copy()
    df_trades[1:] = df_trades.diff()*1000
    #holdings = df_trades.cumsum(axis=0)
    return df_trades

def test_code():

 impact= 0.
 comission = 0 #9.95
 sv=100000
 for insample in [1,0]:
    if insample==1:
     sd=dt.datetime(2008,1,1)
     ed=dt.datetime(2009,12,31)
     prefix="InSample"
    else:
     sd=dt.datetime(2010,1,1)
     ed=dt.datetime(2011,12,31)
     prefix = "OutSample"
    df_benchmark = BestPossibleStrategy.benchmarkPolicy(['JPM'], sd, ed, sv=sv)
    df_trades = testPolicy(['JPM'], sd, ed, sv=sv)

    port_benchmark = compute_portvals(df_benchmark,start_val = sv, commission=comission, impact=impact)
    port_manual = compute_portvals(df_trades, start_val=sv, commission=comission, impact=impact)

    cv_benchmark, dr_benchmark, std_dr_benchmark, sharpe_benchmark=compute_portfolio_stats(port_benchmark.iloc[:,0])
    cv_manual, dr_manual, std_dr_manual, sharpe_manual=compute_portfolio_stats(port_manual.iloc[:,0])

    print
    print "Symbols:", "JPM"
    print "Start Date:", sd
    print "End Date:", ed
    print "Start Value:", sv
    print "Impact:", impact
    print "Commission:", comission

    print prefix+" Benchmark final portfolio =\t"+str(port_benchmark.iloc[-1,0])
    print prefix+" ManualStrategy final portfolio =\t"+str(port_manual.iloc[-1,0])

    print prefix+" Benchmark cv =\t"+str(cv_benchmark)
    print prefix+" ManualStrategy cv =\t"+str(cv_manual)

    print prefix+" Benchmark dr =\t"+str(dr_benchmark)
    print prefix+" ManualStrategy dr =\t"+str(dr_manual)

    print prefix+" Benchmark std_dr =\t"+str(std_dr_benchmark)
    print prefix+" ManualStrategy std_dr =\t"+str(std_dr_manual)

    print prefix+" Benchmark sharpe ratio =\t"+str(sharpe_benchmark)
    print prefix+" ManualStrategy sharpe ratio =\t"+str(sharpe_manual)
    '''
    df = pd.concat([port_benchmark, port_manual],
                   keys=['Benchmark', "Manual"], axis=1)
    df.columns = ['Benchmark', "Manual"]
    df=df/df.iloc[0,:]
    vline = df_trades.copy()
    vline=vline["JPM"][vline["JPM"]!=0]
    holdingSign = np.sign(vline.cumsum())
    vline[:]="black"
    vline.values[holdingSign>0]="green"
    vline.values[holdingSign<0]="red"
    vline=vline[vline!="black"]
    if insample:
        title= "Figure 7:   "+prefix+" Portfolio of Manual Strategy"
    else:
        title = "Figure 8:   " + prefix + " Portfolio of Manual Strategy"
    plot_df(df,title=title,color=["blue", "black"],style=["-","-"],\
            figName="Portfolio_Manual_"+prefix+".png",vline=vline)
    '''

if __name__ == "__main__":
    test_code()
