# name: Yu Bai; ID: ybai67

"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random as rand
from indicators import *
import QLearner as ql
from marketsimcode import *

# name: Yu Bai; ID: ybai67

class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = 1 #placeholder
        self.thresholds={} #{"bbp":[] } # num states per indicator = len([])+1; self.thresholds["holding"]=["long","short","none"]
        #@volitality for experiment 2???
        self.states=[]
        self.cash=[]
        self.holdings=[]
        self.portfolios=[]
        self.sv=0
        self.standarize=False
        self.rar=0.95
        self.converge=0.01
        if self.verbose:
            self.debug = []
            self.action=[]
            self.indicator=np.array([])
    # this method should create a QLearner, and train it for trading

    #@ exit too quick: how to make less conseuctive trading?
    #@ min #states: combination of indicator for 1 state??
    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):

        prices = getPrice(syms=[symbol], sd=sd, ed=ed, includeSPY=True, extendPrior=20)

        # calculate indicators (better standarize as the min/max will be consistent for any data)
        self.standarize=False
        prices,indicators=cal_indicators4strategyLearner(sd,prices,lookback=20,standarize=self.standarize, verbose=self.verbose)
        if self.verbose: indicators.to_csv("JPM_train_indicators.csv")
        self.sv = sv
        indicators = indicators.ix[:,["p2sma","bbp","rsi","momentum","std"]]

        ### model building, skip in testPolicy
        # discretize indicators;
        # try no standarization and cutoff for original scale: unitless & train & test should be more
        if self.standarize:
         self.thresholds["p2sma"]=self.quantile2Thresholds(indicators.ix[:,"p2sma"],[0.1,0.9]) #0.25,0.75
         self.thresholds["bbp"]=self.quantile2Thresholds(indicators.ix[:,"bbp"],[0.01,0.99])
         self.thresholds["rsi"]=self.quantile2Thresholds(indicators.ix[:,"rsi"],[0.12,0.85])
         self.thresholds["momentum"]=self.quantile2Thresholds(indicators.ix[:,"momentum"],[0.85]) #[0.125,0.2,0.75,0.85]
         self.thresholds["std"] = self.quantile2Thresholds(indicators.ix[:, "std"], [0.69]) #[0.65]
        else:
         self.thresholds["p2sma"] = [0.9,1.1] #0.95, 1.05
         self.thresholds["bbp"] = [-0., 1.]
         self.thresholds["rsi"] = [35, 60]
         self.thresholds["momentum"] = [0.2]
         self.thresholds["std"] = [0.07] #0.065
        self.thresholds["holding"] = [-1000.0, 0.0, 1000.0]

        #set self.states
        num_states= (len(self.thresholds["p2sma"])+1)* \
                    (len(self.thresholds["bbp"])+1) * \
                    (len(self.thresholds["rsi"])+1) * \
                    (len(self.thresholds["momentum"])+1)* \
                    (len(self.thresholds["std"])+1) * 3

                   #*1 #for sma_cross
        self.states=np.array(range(0,num_states)).reshape((len(self.thresholds["p2sma"])+1,
                                                           len(self.thresholds["bbp"])+1,
                                                           len(self.thresholds["rsi"])+1, len(self.thresholds["momentum"])+1,
                                                           len(self.thresholds["std"])+1,3 ))
        if self.verbose: print "# states=" + str(num_states)
        # instantize Qlearner with num_state, num_actions & others (all possible states considered even some may not encounter in training data)
        num_actions=3
        self.learner = ql.QLearner(num_states, num_actions,alpha=0.2, gamma=0.5, rar = self.rar, \
                                    radr = 0.999, \
                                    dyna = 0, \
                                    verbose = False)
        if self.verbose: self.printQ("Q_notrain.csv")
        ###\\model building, skip in testPolicy
        self.indicator=np.array([np.append([0,0,0,0,0],indicators.values[0, :])])

        maxepoch=20 # change to ensure time
        nepoch=0
        oldcr=-1000
        cr=0.0001
        while (not self.isConverged(cr,oldcr,limit=self.converge)) and nepoch<maxepoch:
            nepoch = nepoch + 1
            self.initStrategyLearner(prices)
            # state=day1 state=day 1 indicator (+previous day action-induced holding=0)
            #state = self.encodeState(pd.DataFrame([indicators.ix[0,:]],columns=indicators.columns),holding=None)
            state = self.encodeState(pd.DataFrame([indicators.ix[0,:]],columns=indicators.columns),holding=0)
            # action=day1 action = querysetstate(day1 state)
            action = self.learner.querysetstate(state)
            # takeAction=updateRecord(day1 state, day 1 action)=portfolio @day1 after action+ record action in trade df
            self.takeAction(0, prices, action, self.impact)
            if self.verbose:
                self.debug = self.debug + [state]
                self.action = self.action + [action]
                self.indicator = np.append(self.indicator,[np.append([0, self.holdings[0], self.cash[0],prices.values[0,0],0],indicators.values[0, :])],axis=0)

            for whichdate in range(1,prices.shape[0]):
                # newstate=day2 state = day2 indicator (+ previous day action->holding) aka current indicator+current holding
                #newstate = self.encodeState(pd.DataFrame([indicators.ix[whichdate, :]], columns=indicators.columns),holding=None)
                newstate = self.encodeState(pd.DataFrame([indicators.ix[whichdate,:]],columns=indicators.columns),holding=self.holdings[whichdate-1])
                # r = calR(day1 state, day1 action,day2 indicator)= value(day2 state) - value(day1 state) - impact
                if whichdate==1:
                    #self.cash@previous date included impact
                    r = ((prices.values[whichdate,:]*self.holdings[whichdate-1]+self.cash[whichdate-1]) / \
                        (prices.values[whichdate-1,:]*0+self.sv)-1)[0]
                else:
                    r = ((prices.values[whichdate,:]*self.holdings[whichdate-1]+self.cash[whichdate-1]) / \
                        (prices.values[whichdate-1,:]*self.holdings[whichdate-2]+self.cash[whichdate-2])-1)[0]

                # action = day2 action=self.learner.query(day2 state,r) #update action to new action
                action = self.learner.query(newstate,r)
                # ta keAction=updateRecord(day2 state, day 2 action)=portfolio @day1 after action+ record action in trade df
                self.takeAction(whichdate, prices, action, self.impact)
                # state = newstate  #update state var to new state
                state = newstate

                if self.verbose:
                    self.debug = self.debug + [state]
                    self.action = self.action + [action]
                    self.indicator = np.append(self.indicator, [np.append([whichdate,self.holdings[whichdate],self.cash[whichdate],prices.values[whichdate,0],r],indicators.values[whichdate, :])], axis=0)

            if self.verbose: print "rar="+str(self.learner.getRar())
            self.portfolios = self.calPortfolio(self.holdings,prices,self.cash)
            oldcr = cr
            cr = (self.portfolios.ix[-1,0]/self.portfolios.ix[0,0])-1
            if self.verbose: print "epoch="+str(nepoch)+", train cr =" + str(cr)
        if self.verbose:
            from collections import Counter
            print Counter(self.debug)
            self.indicator = self.indicator[1:, :]

    # this method should use the existing policy and test it against new data
    # testPolicy, make sure states matching
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):
        prices = getPrice(syms=[symbol], sd=sd, ed=ed, includeSPY=True, extendPrior=20)
        prices,indicators=cal_indicators4strategyLearner(sd,prices,lookback=20,standarize=self.standarize, verbose=False)
        #if self.verbose: indicators.to_csv("JPM_test_indicators.csv")
        indicators = indicators.ix[:,["p2sma","bbp","rsi","momentum","std"]]
        self.sv=sv
        # optimal policy obtained by finishing training, learner rar set to 0 here even if it's not
        self.learner.setRar(0)

        self.initStrategyLearner(prices)
        #state = self.encodeState(pd.DataFrame([indicators.ix[0, :]], columns=indicators.columns), holding=None)
        state = self.encodeState(pd.DataFrame([indicators.ix[0, :]], columns=indicators.columns), holding=0)
        action = self.learner.querysetstate(state)
        self.takeAction(0, prices, action, self.impact)
        for whichdate in range(1, prices.shape[0]):
            #newstate = self.encodeState(pd.DataFrame([indicators.ix[whichdate, :]], columns=indicators.columns),holding=None) #self.holdings[whichdate-1]
            newstate = self.encodeState(pd.DataFrame([indicators.ix[whichdate, :]], columns=indicators.columns),holding=self.holdings[whichdate-1]) #self.holdings[whichdate-1]
            action = self.learner.querysetstate(newstate)
            self.takeAction(whichdate, prices, action, self.impact)
            state = newstate
        if self.verbose:
            self.portfolios = self.calPortfolio(self.holdings, prices, self.cash)
            cr = (self.portfolios.ix[-1,0]/self.portfolios.ix[0,0])-1
            print "test cr=" +str(cr)
        df_holdings = pd.DataFrame(self.holdings,index=prices.index, columns=prices.columns)
        df_trades = df_holdings.copy()
        df_trades[1:] = df_trades.diff()
        if self.verbose:
            from collections import Counter
            print Counter(self.debug)
        return df_trades


    def initStrategyLearner(self, prices):
        self.holdings = np.zeros(prices.shape[0])
        self.cash = np.zeros(prices.shape[0])
        #self.learner.setRar(self.rar)

    #use quantile; otherwise the input values = thresholds no need function value2Thresholds
    def quantile2Thresholds(self,indicator, quantiles):
        thresholds = indicator.quantile(quantiles).values
        #thresholds=np.insert(thresholds,0,indicator.min())
        #thresholds=np.append(thresholds,indicator.max()+0.1)
        return thresholds

    # encode the a tuple of indicators to a state
    # (if with long/short/exit status: more states, diff action can lead to diff state, R associated with state, not action)
    # (if without long/short/exit status: simpler model/less state/easier converge; next state indep of action, R associated with action not state (holding obsorbed into R))
    def encodeState(self, indicatorVals, holding=None):
        #indicatorVals = df with indicator type as column: pd.DataFrame([[val1,val2...]],columns=["label1","label2"])
        ndigits = indicatorVals.shape[1]
        bins=()
        for each in indicatorVals.columns:
            if each=="momentum":
                bins = bins + (np.asscalar(np.digitize(abs(indicatorVals.ix[0,each]),self.thresholds[each])),)
            elif each=="sma_cross":
                bins = bins + (int(indicatorVals.ix[0,each]),)
            else:
                bins = bins + (np.asscalar(np.digitize(indicatorVals.ix[0,each],self.thresholds[each])),)
        if holding is not None:
            bins = bins + tuple([self.thresholds["holding"].index(holding)])
        #test=self.states[bins]
        return self.states[bins]


    def takeAction(self,whichdate,prices,action,impact):  # update porforlio/holdings; irrelevant with Qlearner updates
        if whichdate==0:    #day1, priorholding =0
            priorholding=0
            priorcash=self.sv
        else:
            priorholding = self.holdings[whichdate-1]
            priorcash = self.cash[whichdate - 1]
        if action ==0:  #go long if possible
            position=1
        elif action==1: #go short if possible
            position=-1
        else:    #exit/cash
            position=0
        updates= updateOneDay(position,prices.ix[whichdate,0],priorholding,priorcash,impact)
        self.holdings[whichdate]=updates[0]
        self.cash[whichdate] = updates[1]
        #self.portfolios = self.holdings * prices + self.cash


    def calPortfolio(self,holdings, prices,cash):
        return holdings.reshape(prices.shape[0],1) * prices + cash.reshape(prices.shape[0],1)

    def isConverged(self,cr, oldcr,limit=0.01):
        if abs((cr-oldcr)/oldcr)<= limit:
            return True
        else:
            return False

    def setConverge(self,value):
        self.converge=value

    def printQ(self, file):
        pd.DataFrame(self.learner.getQ()).to_csv(file)

    def printTrace(self, file):
        first=pd.DataFrame(np.append(np.array(self.debug).reshape(len(self.debug),1), np.array(self.action).reshape(len(self.action),1),axis=1))
        second=pd.DataFrame(self.indicator)
        third=pd.concat([first, second],axis=1)
        third.columns=["state","action","date","holding","cash","price","r","p2sma","bbp","rsi","momentum","std"]
        third.to_csv(file)

    def author(self):
            return 'ybai67'

