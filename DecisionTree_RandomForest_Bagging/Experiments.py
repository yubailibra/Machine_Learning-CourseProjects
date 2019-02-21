#the code shall be added into testlearner.py
import math
import numpy as np
import BagLearner as bl
import math
import sys
import DTLearner as dt
import RTLearner as rt
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import AutoMinorLocator

def testTreeLearnerByLeafSize(learnerClass, leafsizes, trainX, trainY, testX, testY,rounds=None):
    learners = []
    for each_leaf_size in leafsizes:
        kwargs={"leaf_size":each_leaf_size}
        learners.append(learnerClass(**kwargs))

    return getStats(learners,trainX, trainY, testX, testY, rounds)

def testBagLearnerByLeafSize(learnerClass, leafsizes, trainX, trainY, testX, testY,bags=20, rounds=None,boost=False):
    print learnerClass
    learners = []
    for each_leaf_size in leafsizes:
        kwargs = {"learner":learnerClass,"kwargs":{"leaf_size": each_leaf_size},"bags":bags,"boost":boost}
        learners.append(bl.BagLearner(**kwargs))
    return getStats(learners, trainX, trainY, testX, testY,rounds)

def getOneLearnerStats(learner, trainX, trainY, testX, testY, rounds=None):
    if (isinstance(learner, rt.RTLearner) or isinstance(learner, bl.BagLearner)) and rounds is not None:
        rsmeTrain=[]
        rsmeTest=[]
        corrTrain=[]
        corrTest=[]
        traintime=[]
        querytime=[]
        for r in range(rounds):
            t0 = time.time()
            learner.addEvidence(trainX, trainY)  # train it
            traintime.append(time.time()-t0)
            train_predY = learner.query(trainX)  # get the predictions
            x=math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
            rsmeTrain.append(math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0]))

            t0 = time.time()
            test_predY = learner.query(testX)  # get the predictions
            querytime.append(time.time()-t0)
            rsmeTest.append(math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0]))

            corrTrain.append(np.corrcoef(train_predY, y=trainY)[0, 1])
            corrTest.append(np.corrcoef(test_predY, y=testY)[0, 1])
        print str(learner) + " test RMSE std= " + str(np.array(rsmeTest).std())
        return np.array(rsmeTrain).mean(),np.array(rsmeTest).mean(),np.array(corrTrain).mean(),np.array(corrTest).mean(), \
               np.array(traintime).mean(),np.array(querytime).mean(),np.array(traintime).std(),np.array(querytime).std()
    else:
        t0 = time.time()
        learner.addEvidence(trainX, trainY)  # train it
        traintime=(time.time() - t0)

        train_predY = learner.query(trainX)  # get the predictions
        rsmeTrain=(math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0]))

        t0 = time.time()
        test_predY = learner.query(testX)  # get the predictions
        querytime=(time.time() - t0)
        rsmeTest=(math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0]))

        corrTrain=(np.corrcoef(train_predY, y=trainY)[0, 1])
        corrTest=(np.corrcoef(test_predY, y=testY)[0, 1])
        return rsmeTrain,rsmeTest,corrTrain,corrTest,traintime,querytime,0,0


def getStats(learners, trainX, trainY, testX, testY, rounds=None):
    rsmeTrain=[]
    rsmeTest=[]
    corrTrain=[]
    corrTest=[]
    traintime=[]
    querytime=[]
    trainstd=[]
    querystd=[]
    for learner in learners:
        one_rsmeTrain,one_rsmeTest,one_corrTrain,one_corrTest,onetraintime,onequerytime,onetrainstd,onequerystd=\
            getOneLearnerStats(learner, trainX, trainY, testX, testY, rounds)
        rsmeTrain.append(one_rsmeTrain)
        rsmeTest.append(one_rsmeTest)
        corrTrain.append(one_corrTrain)
        corrTest.append(one_corrTest)
        traintime.append(onetraintime)
        querytime.append(onequerytime)
        trainstd.append(onetrainstd)
        querystd.append(onequerystd)
    return np.array(rsmeTrain), np.array(rsmeTest), np.array(corrTrain), np.array(corrTest),\
           np.array(traintime),np.array(querytime),np.array(trainstd),np.array(querystd)


if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    #np.random.shuffle(data)
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    train_scale =trainY.max()-trainY.min()
    test_scale =testY.max()-testY.min()


# Q1
    '''
    np.random.seed(46) #46
    leafsizes=np.arange(1,101)
    nrounds=10
    rsmeTrain_DT, rsmeTest_DT, corrTrain_DT, corrTest_DT, train_DT,query_DT,trainstd_DT,querystd_DT=\
        testTreeLearnerByLeafSize(dt.DTLearner,leafsizes,trainX, trainY, testX, testY,rounds=None)
    rsmeTrain_RT, rsmeTest_RT, corrTrain_RT, corrTest_RT,train_RT,query_RT,trainstd_RT,querystd_RT =\
        testTreeLearnerByLeafSize(rt.RTLearner,leafsizes,trainX, trainY, testX, testY,rounds=nrounds)
    df=pd.DataFrame({"leaf_size":leafsizes,
                     "DTLearner_Training":rsmeTrain_DT/train_scale, #/trainX.shape[0],
                     "DTLearner_Testing":rsmeTest_DT/test_scale,#/testX.shape[0],
                     "RTLearner_Training":rsmeTrain_RT/train_scale,#/trainX.shape[0],
                    "RTLearner_Testing":rsmeTest_RT/test_scale}) #,/testX.shape[0]})
    #ax = df.plot(x="leaf_size",y=["DTLearner_Training","DTLearner_Testing","RTLearner_Training","RTLearner_Testing"],
    #             title="RMSE of DTLearner & RTLearner vs. Leaf size")

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,5))
    df.plot(x="leaf_size", y=["DTLearner_Training","DTLearner_Testing"],ax=axes[0])
    df.plot(x="leaf_size", y=["RTLearner_Training", "RTLearner_Testing"], ax=axes[1])
    axes[0].set_xlabel("Leaf size")
    axes[0].set_ylabel("RMSE (normalized by Y range)")
    #axes[0].set_ylim(0,0.14)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    #    axes[0].set_xscale("log", nonposx='clip')
    axes[1].set_xlabel("Leaf size")
    axes[1].set_ylabel("RMSE (normalized by Y range)")
    axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))

#    axes[1].set_xscale("log", nonposx='clip')
    #fixme need add minor ticks
    fig.subplots_adjust(left=0.1,wspace=0.08)
    fig.savefig("Q1_RTround"+str(nrounds)+"."+sys.argv[1]+".png")
    '''

    #Q2

    np.random.seed(100) #46
    leafsizes=np.arange(1,51)
    nrounds=10
    nbags=20
    
    t0 = time.time()
    rsmeTrain_DT, rsmeTest_DT, corrTrain_DT, corrTest_DT,train_DT,query_DT,trainstd_DT,querystd_DT =\
        testBagLearnerByLeafSize(dt.DTLearner,leafsizes,trainX, trainY, testX, testY,bags=nbags,rounds=nrounds)
    t1= time.time()
    print t1-t0
    t0 = time.time()
    rsmeTrain_RT, rsmeTest_RT, corrTrain_RT, corrTest_RT,train_RT,query_RT,trainstd_RT,querystd_RT =\
        testBagLearnerByLeafSize(rt.RTLearner,leafsizes,trainX, trainY, testX, testY,bags=nbags,rounds=nrounds)
    t1= time.time()
    print t1-t0
    df = pd.DataFrame({"leaf_size": leafsizes,
                       "Bagged.DT_Training": rsmeTrain_DT / train_scale,
                       "Bagged.DT_Testing": rsmeTest_DT / test_scale,
                       "Bagged.RT_Training": rsmeTrain_RT / train_scale,
                       "Bagged.RT_Testing": rsmeTest_RT / test_scale})
    df2 = pd.DataFrame({"leaf_size": leafsizes,
                   "Train.Time_DTlearner": train_DT ,
                   "Train.Time_RTlearner": train_RT,
                   "Train.STD_DTlearner": trainstd_DT,
                   "Train.STD_RTlearner": trainstd_RT})

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))
    df.plot(x="leaf_size", y=["Bagged.DT_Training", "Bagged.DT_Testing"], ax=axes[0])
    df.plot(x="leaf_size", y=["Bagged.RT_Training", "Bagged.RT_Testing"], ax=axes[1])
    axes[0].set_xlabel("Leaf size")
    axes[0].set_ylim(0, 0.14)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    #axes[0].set_xscale("log", nonposx='clip')
    axes[0].set_ylabel("RMSE (normalized by Y range)")
    axes[1].set_xlabel("Leaf size")
    axes[1].set_ylabel("RMSE (normalized by Y range)")
    axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    #axes[1].set_xscale("log", nonposx='clip')
    fig.subplots_adjust(left=0.1, wspace=0.08)
    if nrounds is None:
        fig.savefig("Q2_bag"+str(nbags)+".png")
    else:
    #    fig.savefig("Q2_bag" + str(nbags) + "_avg"+str(nrounds)+"."+sys.argv[1]+".png")
        fig.savefig("Q2_bag" + str(nbags) + "_avg" + str(nrounds) + "seed100.png")

    '''
    fig,ax=plt.subplots(figsize=(4, 5),nrows=1,ncols=1)
    plt.errorbar(x=df2["leaf_size"], y=df2["Train.Time_DTlearner"],yerr=df2["Train.STD_DTlearner"],marker='o',ms=4,color="m",label="Bagged.DT Training Time")
    plt.errorbar(x=df2["leaf_size"], y=df2["Train.Time_RTlearner"], yerr=df2["Train.STD_RTlearner"], marker='o',ms=4,color="c",label="Bagged.RT Training time")
#    df2.plot(x="leaf_size", y=["Train.Time_RTlearner"],yerr="Train.STD_RTlearner",ax=ax,marker='o',ms=4)
    labels, handles = ax.get_legend_handles_labels()
    plt.legend(labels,handles)
    ax.set_xscale("log", nonposx='clip')
    ax.set_xlabel("leaf size")
    ax.set_ylabel("Time eclipse (second)")
    fig.subplots_adjust(left=0.2)
    fig.savefig("Q2_bag" + str(nbags) + "."+sys.argv[1]+"_time.png")
    '''
    '''
    #Q3
    np.random.seed(46) #46
    leafsizes=np.arange(1,51)
    nrounds=10
    deltaT_DT=[]
    deltaT_RT=[]
    rsmeTest_DT=[]
    rsmeTest_RT=[]
    rsmeTrain_DT, rsmeTest_DT, corrTrain_DT, corrTest_DT,train_DT,query_DT,trainstd_DT,querystd_DT = \
            testTreeLearnerByLeafSize(dt.DTLearner, leafsizes, trainX, trainY, testX, testY,nrounds)
    rsmeTrain_RT, rsmeTest_RT, corrTrain_RT, corrTest_RT,train_RT,query_RT,trainstd_RT,querystd_RT = \
        testTreeLearnerByLeafSize(rt.RTLearner, leafsizes, trainX, trainY, testX, testY,rounds=nrounds)
    df = pd.DataFrame({"leaf_size": leafsizes,
                       "DT_TestRMSE": rsmeTest_DT/test_scale,
                       "DT_TrainingTime": train_DT,
                       "RT_TestRMSE": rsmeTest_RT/test_scale,
                       "RT_TrainingTime": train_RT,
                       "DT_TestCorrelation":corrTest_DT,
                       "RT_TestCorrelation":corrTest_RT})
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(10, 5))
    df.plot(x="leaf_size", y=["DT_TestRMSE", "RT_TestRMSE"], ax=axes[0],color=["m","c"])
  #  df.plot(x="leaf_size", y=["DT_TestCorrelation", "RT_TestCorrelation"], ax=axes[1],color=["m","c"])
    df.plot(x="leaf_size", y=["DT_TrainingTime", "RT_TrainingTime"], ax=axes[1],color=["m","c"])
    axes[0].set_xlabel("Leaf size")
    axes[0].set_ylabel("RMSE (normalized to Y range)")
    #axes[1].set_xlabel("Leaf size")
    #axes[1].set_ylabel("Correlation coefficient")
    axes[1].set_xlabel("Leaf size")
    axes[1].set_ylabel("Time elapse (s)")
    axes[1].set_xscale("log", nonposx='clip')
    axes[0].set_ylim(0.05,0.14)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    #axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    #axes[2].xaxis.set_minor_locator(AutoMinorLocator(5))

    fig.subplots_adjust(wspace=0.5)
    fig.savefig("Q3.png")

    '''
    '''
    #new Q3:
    np.random.seed(46) #46
    bestleaf = 10
    if sys.argv[1]=="3_groups.csv":
        bestleaf=10
    elif sys.argv[1]=="ripple.csv":
        bestleaf=2
    elif sys.argv[1]=="Istanbul_mine.csv":
        bestleaf=9
    elif sys.argv[1]=="winequality-red.csv":
        bestleaf=35
    elif sys.argv[1]=="winequality-white.csv":
        bestleaf=60
    leafsizes=[bestleaf]
    nrounds=10
    rsmeTrain_DT, rsmeTest_DT, corrTrain_DT, corrTest_DT, train_DT,query_DT,trainstd_DT,querystd_DT=\
        testTreeLearnerByLeafSize(dt.DTLearner,leafsizes,trainX, trainY, testX, testY,rounds=nrounds)
    rsmeTrain_RT, rsmeTest_RT, corrTrain_RT, corrTest_RT,train_RT,query_RT,trainstd_RT,querystd_RT =\
        testTreeLearnerByLeafSize(rt.RTLearner,leafsizes,trainX, trainY, testX, testY,rounds=nrounds)
    print "\nDT_TestRMSE"
    print rsmeTest_DT/test_scale
    print "DT_trainTime"
    print train_DT
    print "DT_trainStd"
    print trainstd_DT

    print "RT_TestRMSE"
    print rsmeTest_RT/test_scale
    print "RT_trainTime"
    print train_RT
    print "RT_trainStd"
    print trainstd_RT
    print "\n"
    '''
    '''
    df=pd.DataFrame({"DT_TestRMSE":[0.1946374,0.2379575,0.09158098,0.13910427,0.1520382],
                     "DT_TrainingTime":[0.02572393,0.09197211,0.02948499,0.02439618,0.03539419],
                     "DT_TrainingSTD":[0,0,0,0,0],
                     "RT_TestRMSE": [0.20797334, 0.25902199,0.10456762,0.15254787,0.15769008],
                     "RT_TrainingTime": [0.00994055,0.053863,0.00830548,0.0076386,0.01626284],
                     "RT_TrainingSTD": [0.00089279,0.00139418,0.00075457,0.00039153,0.00067762]
                     },
                    index=["3_groups","ripple","Istanbul","winequality-red","winequality-white"])
    print df.loc[:,["DT_TrainingSTD","RT_TrainingSTD"]]

    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,5),sharey=False)
    df.plot.bar(y=["DT_TestRMSE", "RT_TestRMSE"], ax=axes[0], color=["m", "c"],rot=45)
    df.plot.bar(y=["DT_TrainingTime", "RT_TrainingTime"],yerr=df.loc[:,["DT_TrainingSTD","RT_TrainingSTD"]], ax=axes[1], color=["m", "c"],rot=45)
    axes[0].set_ylabel("RMSE (normalized to Y range)")
    axes[1].set_ylabel("Time elapse (s)")
    fig.subplots_adjust(wspace=0.5,bottom=0.25)
    fig.savefig("Q3.png")
    '''