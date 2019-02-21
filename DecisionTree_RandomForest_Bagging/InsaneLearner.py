import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner:
    def __init__(self,verbose=False):
        self.learner=bl.BagLearner(bl.BagLearner,
                                   kwargs={"learner":lrl.LinRegLearner,"kwargs":{},"bags":20}, #"boost":False
                                   bags=20,verbose=verbose) #boost=False

    def addEvidence(self,Xtrain,Ytrain):
        self.learner.addEvidence(Xtrain,Ytrain)

    def query(self,Xtest):
        return self.learner.query(Xtest)


    def author(self):
        return "ybai67"
