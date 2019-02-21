# name: Yu Bai; ID: ybai67

"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

# name: Yu Bai; ID: ybai67

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.rar = rar
        self.radr = radr
        self.dyna=dyna
        self.alpha = alpha
        self.gamma=gamma

        #self.Q = np.array(np.random.randn(num_states*num_actions)).reshape(num_states,num_actions) #CHECK if normal or uniform distribution is good init
        self.Q = np.array(np.random.uniform(-0.000001,0.000001,num_states*num_actions)).reshape(num_states,num_actions)
        self.s = 0
        self.a = 0

        self.experience=[]
        #self.Tc = np.array(np.zeros((num_states, num_actions, num_states)))
        #self.Tc[:,:,:]=0.00001
        #self.T = np.array(np.zeros((num_states, num_actions, num_states)))
        #self.R = np.array(np.zeros((num_states, num_actions)))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        a = self.actionAtState(s)
        self.a = a
        if self.verbose: print "s =", s,"a =",a
        return a

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r:
        @returns: The selected action
        """
        if self.dyna>0:
            self.experience.append((self.s, self.a, s_prime, r)) # CHECK: uniqueness of tuples??s
            #self.buildTR(self.s, self.a, s_prime, r) #if use dyna2

        #updata Q
        self.updateQ(s_prime, r)

        # take action at s_prime
        action = self.querysetstate(s_prime)
        if self.verbose: print "s =", s_prime,"r =",r,"a =",action

        # update randomness
        self.updateRar()

        return action

    def updateQ(self, s_prime, r):
        if self.dyna==0:
            self.Q[self.s, self.a]= (1-self.alpha)*self.Q[self.s, self.a] + \
                    self.alpha * (r + self.gamma * (self.Q[s_prime,:]).max())
        else: # dyna not change current position: self.s, self.a
            self.dyna1()

    def actionAtState(self, s): #no update Q table here
        if rand.uniform(0.0, 1.0) <= self.rar:  # going rogue
            a = rand.randint(0, self.num_actions-1)  # choose the random direction
        else:
            a= np.argmax(self.Q[s,:])
        return a


    def updateRar(self):
        self.rar = self.rar * self.radr
        if self.verbose: print "rar =", self.rar

    def setRar(self,newRar):
        self.rar = newRar
    def getRar(self):
        return self.rar

    def getQ(self):
        return self.Q

    def dyna1(self):
        for i in range(self.dyna):
            whichExperience = np.random.randint(0, len(self.experience), size=1)[0]
            mys, mya, mys_prime, myr = self.experience[whichExperience]
            self.Q[mys, mya] = (1 - self.alpha) * self.Q[mys, mya] + \
                               self.alpha * (myr + self.gamma * (self.Q[mys_prime, :]).max())

    def dyna2(self):
        for i in range(self.dyna):
            mys = np.random.randint(0, self.num_states,size=1)[0]
            mya = np.random.randint(0, self.num_actions,size=1)[0]
            cumu = np.cumsum(self.T[mys,mya,:])
            myrand = rand.uniform(0.0, 1.0)
            whichstair = np.searchsorted(cumu,myrand)

            mys_prime = np.argmax(self.T[mys,mya,:]) #change to sample by prob
            myr = self.R[mys, mya]
            self.Q[mys, mya] = (1 - self.alpha) * self.Q[mys, mya] + \
                               self.alpha * (myr + self.gamma * (self.Q[mys_prime, :]).max())

    def buildTR(self,s, a, s_prime, r):
        self.Tc[s,a,s_prime]=self.Tc[s,a,s_prime]+1
        self.T = self.Tc/self.Tc.sum(axis=2).reshape(self.num_states,self.num_actions,1)
        self.R[s,a]= (1-self.alpha)*self.R[s,a] + self.alpha * r

    def author(self):
        return 'ybai67'


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
