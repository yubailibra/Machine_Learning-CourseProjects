import pandas as pd
import numpy as np
import math

class DTLearner:
    def __init__(self,leaf_size=1,verbose=False):
        #self.data=np.empty((0,0))
        self.leaf_size=leaf_size
        self.verbose=verbose
        self.tree=0

    #training
    def addEvidence(self,Xtrain, Ytrain):
        #@@@ fill NA values, may need care about dates!!!
        # merge X & Y into a single ndarray
        data=np.append(Xtrain, np.reshape(Ytrain,(len(Ytrain),1)), axis=1)
        if np.isnan(np.sum(data)):
            df=pd.DataFrame(data)
            df=df.fillna(method="ffill")
            df=df.fillna(method="bfill")
            data=df.as_matrix()
        self.tree=self.buildTree(data)

    # testing
    def query(self,Xtest):
        return np.array([self.queryOneSample(oneX) for oneX in Xtest ])

    def queryOneSample(self,oneX):
        node = 0 #start from the ROOT
        while(self.tree[node,0]!=-1):
            whichvar=int(self.tree[node,0])
            if oneX[whichvar] <= self.tree[node,1]:
                node = int(node + self.tree[node,2])
            else:
                node = int(node + self.tree[node,3])
        return self.tree[node,1]

    def buildTree(self,indata):
        if(self.verbose):
            print "\ncurrent branch size: "+str(indata.shape[0])
            if(indata.shape[0]==2):
                print indata
        if indata.shape[0] <= self.leaf_size:
            if(self.verbose):
                print "Note: branch size <= leaf_size, aggregated"
            return np.array([[-1,self.aggregateLeaf(indata),np.nan,np.nan]]) # -1 for leaf
        if np.round(np.std(indata[:,-1],ddof=1),decimals=14) == 0: #this may violate leaf_size requirement so better skip??
            if(self.verbose):
                print "Note: branch shares common Y, aggregated (branch size="+str(indata.shape[0])+")"
            return np.array([[-1,indata[0,-1],np.nan, np.nan]])
        else:
            indexi=self.findBestFeature(indata) #indexi used as label for the factor
            SplitVal=np.median(indata[:,indexi])
            if(self.verbose):
                print "SplitVal = "+str(SplitVal)
            leftdata = indata[indata[:,indexi]<=SplitVal]
            rightdata = indata[indata[:,indexi]>SplitVal]
            if leftdata.shape[0] == 0 or rightdata.shape[0] == 0: #no further split, this will take care of std=0 => invalid corr
                #@ someone suggest to even split if down to 2 rows but this may cause bias in query (mean Y is better)
                #if indata.shape[0]==2:
                if(self.verbose):
                    print "Note: no further split, aggregated (branch size="+str(indata.shape[0])+")"
                return np.array([[-1,self.aggregateLeaf(indata),np.nan,np.nan]])
            else:
                left = self.buildTree(leftdata) #check if split by ROW; should be
                right = self.buildTree(rightdata)
                root = np.array([[indexi, SplitVal, 1, left.shape[0]+1]])
                tree = np.concatenate((root,left, right),axis=0)
                if(self.verbose):
                    print "current tree size = ("+str(tree.shape[0]) +" "+str(tree.shape[1])+"); "+\
                       "root size=("+str(root.shape[0])+" "+str(root.shape[1])+"); "+\
                       "left size=("+str(left.shape[0])+" "+str(left.shape[1])+"); "+ \
                       "right size=(" + str(right.shape[0]) + " " + str(right.shape[1]) + ");"
                return tree

    def findBestFeature(self,indata):
        #use max(abs(PCC)) to select best var to split on
        #abs_coefficients = map(math.fabs, self.getPCC(indata))
        whichBestVar=np.argmax(self.getPCC(indata))
        if(self.verbose):
            print "best variable selected: " + str(whichBestVar)
        return whichBestVar

    # compute pearson correlation coefficients between 0:N-1 & -1 column of indata
    def getPCC(self, indata):
        #check if Y or any X has 0 standard deviation that will produce invalid correlation coefficients
        col_wise_std=np.round(np.std(indata,axis=0, ddof=1),decimals=14)
        good_index=col_wise_std!=0
        if False in good_index:
            # Y no change, this shall not happen actually as filtered above;or all X no change
            if(not(good_index[indata.shape[1]-1]) or not(True in good_index[0:-1])):
                return [0]*(indata.shape[1]-1)
            pre_coefficients=np.absolute(np.round(np.corrcoef(indata[:,good_index], rowvar=0)[0:-1, -1],decimals=14))
            coefficients = np.array([0.0]*(indata.shape[1]-1))
            np.place(coefficients, good_index[0:-1], pre_coefficients)
            x=0
        else:
            coefficients=np.absolute(np.round(np.corrcoef(indata, rowvar=0)[0:-1, -1],decimals=14))
        if(self.verbose):
            print "abs correlation coefficients:"
            print coefficients
        return coefficients


    #in case more complicated aggregation needed
    def aggregateLeaf(self,indata):
        return np.mean(indata[:,-1])

    def author(self):
        return 'ybai67'


if __name__ == "__main__":
    mylearner=DTLearner()
    print mylearner.data.shape
    mylearner.addEvidence((np.random.normal(0,0.1,(10,3))),
                          (np.random.normal(0,0.1,size=10)))
    print mylearner.data.shape
    #mylearner.query(pd.DataFrame(np.random.normal(0,0.1,(10,1))));
    print mylearner.aggregateLeaf(mylearner.data)
    print mylearner.author()
