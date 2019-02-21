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

def test_code():
    import time
    comission = 0
    sv = 100000
    symbol = "JPM"
    robot_qlearning_testing_seed = 14810900
    rand.seed(robot_qlearning_testing_seed)
    np.random.seed(robot_qlearning_testing_seed)

    impacts=[0,0.005,0.01,0.02]
    ntrades=np.zeros(4).reshape(1,4)
    incvs=np.zeros(4).reshape(1,4)
    indrs = np.zeros(4).reshape(1, 4)
    insds = np.zeros(4).reshape(1, 4)
    insrs = np.zeros(4).reshape(1, 4)

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    print "Symbols:", symbol
    print "Start Date:", sd
    print "End Date:", ed


    for r in range(3):
      r_trade=[]
      r_incv=[]
      r_indr=[]
      r_insd=[]
      r_insr=[]
      for impact in impacts:
            print  "-------------------------"
            print "round "+ str(r) + " impact=" + str(impact)
            learner=sl.StrategyLearner(False,impact)
            learner.setConverge(0.001)
            prefix = "InSample"
            start_time = time.time()
            learner.addEvidence(symbol, sd, ed, sv)
            df_trades = learner.testPolicy(symbol, sd, ed, sv)
            port_learner = compute_portvals(df_trades, start_val=sv, commission=comission, impact=impact)
            cv_learner, dr_learner, std_dr_learner, sharpe_learner = compute_portfolio_stats(port_learner.iloc[:, 0])
            junk = df_trades[df_trades["JPM"] != 0]
            r_trade = r_trade + [junk.shape[0]]
            r_incv = r_incv + [cv_learner]
            r_indr = r_indr + [dr_learner]
            r_insd = r_insd + [std_dr_learner]
            r_insr = r_insr + [sharpe_learner]

            #print prefix + " StrategyLearner final portfolio =\t" + str(port_learner.iloc[-1, 0])
            print prefix + " StrategyLearner cv =\t" + str(cv_learner)
            #print prefix + " StrategyLearner dr =\t" + str(dr_learner) #use expr1 dr as reference to argue about impact overwhelms
            #print prefix + " StrategyLearner std_dr =\t" + str(std_dr_learner)
            print prefix + " StrategyLearner sharpe ratio =\t" + str(sharpe_learner)
            print prefix + "# trades=" + str(junk.shape[0])

      ntrades = np.append(ntrades,[r_trade],axis=0)
      incvs = np.append(incvs,[r_incv],axis=0)
      indrs = np.append(indrs,[r_indr],axis=0)
      insds = np.append(insds,[r_insd],axis=0)
      insrs = np.append(insrs,[r_insr],axis=0)
    mean_cv=[]
    sd_cv=[]
    mean_sr=[]
    sd_sr=[]
    mean_ntrade=[]
    sd_ntrade=[]
    for whichimpact in range(4):
        mean_cv = mean_cv +[(incvs[1:, whichimpact]).mean()]
        sd_cv = sd_cv +[(incvs[1:, whichimpact]).std()]
        mean_sr = mean_sr +[(insrs[1:, whichimpact]).mean()]
        sd_sr = sd_sr +[(insrs[1:, whichimpact]).std()]
        mean_ntrade = mean_ntrade +[(ntrades[1:, whichimpact]).mean()]
        sd_ntrade = sd_ntrade +[(ntrades[1:, whichimpact]).std()]

        print "impact="+ str(impacts[whichimpact]) + \
             ":\tcv="+ str((incvs[1:,whichimpact]).mean()) + \
             "\tsharpeRatio="+str((insrs[1:,whichimpact]).mean()) + \
             "\tnTrades=" + str((ntrades[1:, whichimpact]).mean())
             # "\tdailyRaturn="+str((indrs[1:,whichimpact]).mean()) +\
             # "\tstdev_dailyReturn="+str((insds[1:,whichimpact]).mean()) +\
    '''
    mean_cv=[0.79,0.538792333333,0.31638,0.155529333333]#[0.634133333333, 0.488141333333, 0.320209,0.0562953333333]
    sd_cv=[0.2,0.2,0.2,0.2]
    mean_sr = [3.67706970395,2.76300485178,1.86901211398,1.06198885143]#[2.93486859184,2.48597671425,1.79200555181,0.481159734102]
    sd_sr=[0.5,0.5,0.5,0.5]
    mean_ntrade = [99.3333333333,78.3333333333,44.6666666667,19.3333333333]#[67.3333333333,47.0,45.0,18.0]
    sd_ntrade=[5,5,5,5]
    #do a figure
    '''
    df = pd.DataFrame({"impact":impacts,
                       "Cumulative.Return":mean_cv,
                       "Sharpe.Ratio":mean_sr,
                       "num.of.trades":mean_ntrade,
                       "sd_cv":sd_cv,
                       "sd_sr": sd_sr,
                       "sd_ntrade":sd_ntrade})
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(10, 5))
    df.plot(x="impact", y=["Cumulative.Return"],yerr=sd_cv, ax=axes[0],marker="o",color="r")
    axes[0].set_xlim(-0.0005, 0.0205)
    axes[0].set_ylabel("Cumulative Return")
    df.plot(x="impact", y=["Sharpe.Ratio"], yerr=sd_sr,ax=axes[1],marker="o",color="g")
    axes[1].set_xlim(-0.0005, 0.0205)
    axes[1].set_ylabel("Sharpe Ratio")
    df.plot(x="impact", y=["num.of.trades"], yerr=sd_ntrade, ax=axes[2],marker="o",color="b")
    axes[2].set_xlim(-0.0005, 0.0205)
    axes[2].set_ylabel("number of trades")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.suptitle("Figure 3: Performance of StrategyLearner vs. impact")
    fig.savefig("plot_Impact_effects.png")

if __name__ == "__main__":
    test_code()
