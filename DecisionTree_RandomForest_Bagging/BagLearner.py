import pandas as pd
import numpy as np

class BagLearner:
    def __init__(self,learner,kwargs={"leaf_size":1},bags=20, boost=False, verbose=False):
        self.learnerClass = learner # pass in class name, as if defined an internal class
        self.learners=[]
        self.bags=bags
        self.boost=boost
        self.kwargs=kwargs
        self.verbose =verbose
        if not("verbose" in kwargs.keys()):
            kwargs["verbose"]=verbose
        for i in range(0,self.bags):
            self.learners.append(self.learnerClass(**self.kwargs))

    def addEvidence(self,Xtrain, Ytrain):
        counter=0
        if not self.boost:
            for oneLearner in self.learners:
                counter=counter+1
                sampleIndices = np.random.choice(np.arange(Xtrain.shape[0]), size=Xtrain.shape[0], replace=True)
                if self.verbose:
                    print "\n**************************"
                    print str(self.learnerClass) + " " +str(counter)
                    print Xtrain[sampleIndices].shape
                    print Ytrain[sampleIndices].shape
                oneLearner.addEvidence(Xtrain[sampleIndices],Ytrain[sampleIndices])
        else:
            pass



    def query(self,Xtest):
        bagY = [self.learners[0].query(Xtest)]
        for oneLearner in self.learners[1:]:
            bagY = np.concatenate((bagY, [oneLearner.query(Xtest)]),axis=0)
        if len(self.learners)==1:
            return bagY[0]
        return bagY.mean(axis=0)



    def author(self):
        return 'ybai67'
