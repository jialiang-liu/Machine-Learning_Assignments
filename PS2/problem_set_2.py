#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

# load the data
def loadsparsedata(fn):
    
    fp = open(fn,"r")
    lines = fp.readlines()
    maxf = 0;
    for line in lines:
        for i in line.split()[1::2]:
            maxf = max(maxf,int(i))
    
    X = np.zeros((len(lines),maxf))
    Y = np.zeros((len(lines)))
    
    for i, line in enumerate(lines):
        values = line.split()
        Y[i] = int(values[0])
        for j,v in zip(values[1::2],values[2::2]):
            X[i,int(j)-1] = int(v)
    
    return X,Y

#from IPython.display import display, clear_output

def learnlogreg(X,Y01,lam):
    # X is an np.array of shape (m,n) with the first column all 1s
    # Y01 is an np.array of shape (m,1) with element either 0 or 1
    # lam is the regularization strength (a scalar)
    
    Y = Y01.copy() # rules derived in class assume that 
    Y[Y==0] = -1 # the classes are represented by +1 and -1, not +1, and 0
    (m,n) = X.shape

    w = np.zeros((n))
    
    eta = np.sqrt(2.5/lam) # the suggested stepsize (see above)
    
    ### YOUR CODE HERE
    nX = np.mat(X).T
    mXY = -np.multiply(nX,Y)
    c1 = 2*lam/m
    while 1:
        #calculate pi = sigma(yiwTxi)
        pi = 1.0/(1.0+np.exp(w@mXY))
        #calculate gi = -(1-pi)yixi
        gi = np.multiply((1.0-pi),mXY).T
        g = np.squeeze(((gi.sum(axis = 0))*(1/m)).tolist(),axis = 0)+w*c1
        #add regularization, and not apply to first column
        g[0] -= w[0]*c1
        w -= g*eta
        if g@g<1e-10:
            break
    
    return w

(trainX,trainY) = loadsparsedata("/Users/travis/Desktop/CS171/Homework/PS2/spamtrain.txt")
trainX = np.column_stack((np.ones(trainX.shape[0]),trainX)) # add column of 1s as zeroth feature

w = learnlogreg(trainX,trainY,1e-3) # just an example to check with lambda=0.001

def linearerror(X,Y,w):
    # returns error *rate* for linear classifier with coefficients w
    m = Y.shape[0]
    predy = X.dot(w)
    err = (Y[predy>=0]<0.5).sum() + (Y[predy<0]>=0.5).sum()
    return err/m

linearerror(trainX,trainY,w) # should be 0.001333... (if lambda=0.001, above)

lambdas = np.logspace(-3,2,10)

### Your Code Here
#import data
(trainX,trainY) = loadsparsedata("/Users/travis/Desktop/CS171/Homework/PS2/spamtrain.txt")
trainX = np.column_stack((np.ones(trainX.shape[0]),trainX))
ntrainX = np.array_split(trainX,5,0)
ntrainY = np.array_split(trainY,5,0)
(testX,testY) = loadsparsedata("/Users/travis/Desktop/CS171/Homework/PS2/spamtest.txt")
testX = np.column_stack((np.ones(testX.shape[0]),testX))

#calculate the error rate when nth fold is the validation set
def nfold(X,Y,lambdas,n):
    testX = X[n-1]
    testY = Y[n-1]
    X = np.delete(X,n-1,axis = 0)
    X = np.concatenate((X[0],X[1],X[2],X[3]),axis = 0)
    Y = np.delete(Y,n-1,axis = 0)
    Y = np.concatenate((Y[0],Y[1],Y[2],Y[3]),axis = 0)
    error = np.zeros((10))
    for i in range(10):
        error[i] = linearerror(testX,testY,learnlogreg(X,Y,lambdas[i]))
    return error

#calculate error rate for cross validation and regular training
crossrate = np.zeros((5,10))
for n in range(5):
    crossrate[n] = nfold(ntrainX,ntrainY,lambdas,n+1)
crossrate = np.mat(crossrate).T
avgcross = np.mean(crossrate, axis=1)
allrate = np.zeros((10))
for n in range(10):
    allrate[n] = linearerror(testX,testY,learnlogreg(trainX,trainY,lambdas[n]))
    
#plot data
plt.semilogx(lambdas,allrate,label = 'Testing Error Rate')
plt.semilogx(lambdas,avgcross,label = 'Cross Validation Error Rate')
plt.xlabel('λ')
plt.ylabel('Error rates')
plt.legend()
plt.show()

end = time.time()
print(end-start)