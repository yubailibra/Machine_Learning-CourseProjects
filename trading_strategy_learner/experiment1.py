# name: Yu Bai; ID: ybai67

import datetime as dt
import pandas as pd
import util as ut
import random as rand
from indicators import *
import StrategyLearner as sl
import ManualStrategy as ms
from marketsimcode import *



#####################################################################################
def benchmarkPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        prices = getPrice(symbol, sd, ed)
        df_trades = prices.copy()
        df_trades[:] = 0
        df_trades.iloc[0, :] = 1000
        return df_trades

'''
def evalPolicyRef(symbol, student_trades, startval,market_impact,commission_cost):
    import grade_strategy_learner as g
    orders_df = pd.DataFrame(columns=['Shares','Order','Symbol'])
    for row_idx in student_trades.index:
        nshares = student_trades.loc[row_idx][0]
        if nshares == 0:
            continue
        order = 'sell' if nshares < 0 else 'buy'
        new_row = pd.DataFrame([[abs(nshares),order,symbol],],columns=['Shares','Order','Symbol'],index=[row_idx,])
        orders_df = orders_df.append(new_row)
    portvals = g.compute_portvals(orders_df,startval,market_impact,commission_cost)
    return float(portvals[-1]/portvals[0])-1
'''
def test_code():
    import time
    comission = 0
    impact=0
    sv = 100000
    symbol = "JPM"
    robot_qlearning_testing_seed = 14810900
    rand.seed(robot_qlearning_testing_seed)
    np.random.seed(robot_qlearning_testing_seed)

    print  "-------------------------"
    print  "-------------------------"
    print "Symbols:", symbol
    print "Start Value:", sv
    print "Impact:", impact
    print "Commission:", comission

    #cases=getPrice(syms=["ML4T-220","AAPL","SINE_FAST_NOISE","UNH"], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), includeSPY=False, extendPrior=None)
    #cases.to_csv("cases_train.csv")
    learner=sl.StrategyLearner(False,impact)
    learner.setConverge(0.001)
    for insample in [1, 0]:
            if insample == 1:
                sd = dt.datetime(2008, 1, 1)
                ed = dt.datetime(2009, 12, 31)
                prefix = "InSample"
                print "------------ InSample ------------"
                print sd , "--" , ed
                start_time = time.time()
                learner.addEvidence(symbol, sd, ed, sv)
                #print "time elapse=" + str(time.time()-start_time)
                #learner.printQ("Q_train.csv")
                #learner.printTrace("trace_train.csv")
            else:
                sd = dt.datetime(2010, 1, 1)
                ed = dt.datetime(2011, 12, 31)
                prefix = "OutSample"
                print
                print "------------ OutSample ------------"
                print sd , "--" , ed
            df_benchmark = benchmarkPolicy([symbol], sd, ed, sv=sv)
            df_trades = learner.testPolicy(symbol, sd, ed, sv)
            df_manual = ms.testPolicy(symbol, sd, ed, sv)

            port_benchmark = compute_portvals(df_benchmark, start_val=sv, commission=comission, impact=impact)
            port_learner = compute_portvals(df_trades, start_val=sv, commission=comission, impact=impact)
            port_manual = compute_portvals(df_manual, start_val=sv, commission=comission, impact=impact)
            ntrade_manual = df_manual[df_manual.ix[:,0] != 0].shape[0]
            ntrade_learner = df_trades[df_trades.ix[:,0] != 0].shape[0]

            cv_benchmark, dr_benchmark, std_dr_benchmark, sharpe_benchmark = compute_portfolio_stats(
                port_benchmark.iloc[:, 0])
            cv_learner, dr_learner, std_dr_learner, sharpe_learner = compute_portfolio_stats(port_learner.iloc[:, 0])
            cv_manual, dr_manual, std_dr_manual, sharpe_manual = compute_portfolio_stats(port_manual.iloc[:, 0])
            #refcv = evalPolicyRef(symbol, df_trades, sv,impact,0)

            print prefix + " Benchmark final portfolio =\t" + str(port_benchmark.iloc[-1, 0])
            print prefix + " Manual final portfolio =\t" + str(port_manual.iloc[-1, 0])
            print prefix + " StrategyLearner final portfolio =\t" + str(port_learner.iloc[-1, 0])
            print
            print prefix + " Benchmark cv =\t" + str(cv_benchmark)
            print prefix + " Manual final cv =\t" + str(cv_manual)
            print prefix + " StrategyLearner cv =\t" + str(cv_learner)
            print
            print prefix + " Benchmark dr =\t" + str(dr_benchmark)
            print prefix + " Manual dr =\t" + str(dr_manual)
            print prefix + " StrategyLearner dr =\t" + str(dr_learner)
            print
            print prefix + " Benchmark std_dr =\t" + str(std_dr_benchmark)
            print prefix + " Manual std_dr =\t" + str(std_dr_manual)
            print prefix + " StrategyLearner std_dr =\t" + str(std_dr_learner)
            print
            print prefix + " Benchmark sharpe ratio =\t" + str(sharpe_benchmark)
            print prefix + " Manual sharpe ratio =\t" + str(sharpe_manual)
            print prefix + " StrategyLearner sharpe ratio =\t" + str(sharpe_learner)
            print
            print prefix + " Manual: # of trades =\t" + str(ntrade_manual)
            print prefix + " StrategyLearner: # of trades =\t" + str(ntrade_learner)


            df = pd.concat([port_benchmark, port_manual, port_learner],
                           keys=['Benchmark', "ManualStrategy","StrategyLearner"], axis=1)
            df.columns = ['Benchmark',"ManualStrategy", "StrategyLearner"]
            df = df / df.iloc[0, :]

            vline = df_trades.copy()
            vline = vline[symbol][vline[symbol] != 0]
            holdingSign = np.sign(vline.cumsum())
            vline[:] = "black"
            vline.values[holdingSign > 0] = "green"
            vline.values[holdingSign < 0] = "red"
            vline = vline[vline != "black"]
            if insample:
                title = "Figure 1:   " + prefix + " "+symbol+" Portfolio of StrategyLearner"
            else:
                title = "Figure 2:   " + prefix +" "+symbol+ " Portfolio of StrategyLearner"
            plot_df(df, title=title, color=["blue", "orange","black"], style=["-", "-"], \
                    figName="Portfolio_StrategyLearner_" + prefix +"_"+symbol+ ".png", vline=vline)


if __name__ == "__main__":
    test_code()
