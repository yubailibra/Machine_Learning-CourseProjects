import pandas as pd
import numpy as np
import math


class RTLearner:
    def __init__(self,leaf_size=1,verbose=False):
        self.leaf_size=leaf_size
        self.verbose=verbose
        self.tree=0
        #np.random.seed(0)  #ask if need to use seed

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
            indexi, SplitVal, leftdata, rightdata =self.findBestFeature(indata)
            if leftdata.shape[0] == 0 or rightdata.shape[0] == 0: #no further split, this will take care of std=0 => invalid corr
                #@ someone suggest to even split if down to 2 rows but this may cause bias in query (mean Y is better)
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
        #randomly select a var with NON_ZERO_STD to split on
        # if there is no further split, try 10 times and if still no further split then aggregate to a leaf
        col_wise_std=np.std(indata[:,0:-1],axis=0, ddof=1)
        good_index=np.nonzero(col_wise_std)[0]
        if len(good_index)==0:
            indexi=0
            return indexi, indata[0,indexi], indata, np.empty((0,0))
        attempts=0
        while attempts<=9 :
            indexi=np.random.choice(good_index)
            randrows=np.random.choice(range(indata.shape[0]),size=2,replace=False)
            SplitVal = (indata[randrows[0], indexi]+indata[randrows[1], indexi])/2
            #SplitVal = np.median(indata[:,indexi])
            if (self.verbose):
                print "variable selected: " + str(indexi)+" SplitVal = " + str(SplitVal)
            leftdata = indata[indata[:, indexi] <= SplitVal]
            rightdata = indata[indata[:, indexi] > SplitVal]
            if leftdata.shape[0]>0 and rightdata.shape[0]>0:
                break
            attempts=attempts+1
            if(self.verbose and attempts>0):
                print indata
                print "try "+str(attempts)
        if(self.verbose):
            print "final variable selected: " + str(indexi)+" SplitVal = " + str(SplitVal)
        return indexi, SplitVal, leftdata, rightdata


    #in case more complicated aggregation needed
    def aggregateLeaf(self,indata):
        return np.mean(indata[:,-1])

    def author(self):
        return 'ybai67'
