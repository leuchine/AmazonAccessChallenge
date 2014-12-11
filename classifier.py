from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn import cross_validation,metrics
import numpy as np
import random

def sparsify(X, X_test):
    """Return One-Hot encoded datasets."""
    enc = OneHotEncoder()
    enc.fit(np.vstack((X, X_test)))
    return enc.transform(X), enc.transform(X_test)

class OneHotEncoder():
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.
    """
    def __init__(self):
        self.keymap = None

    def fit(self, X):
        self.keymap = []
        for col in X.T:
            li=[]
            for i in col:
                li.append(i)
            uniques = set(li)
            self.keymap.append(dict([(key, i) for i, key in enumerate(uniques)]))

    def transform(self, X):
        if self.keymap is None:
            self.fit(X)

        outdat = []
        for i in X:
            line=[]
            for j,k in enumerate(i):
                line.append(self.keymap[j][k])
            outdat.append(line)
        return outdat


def create_tuples(Z):
    cols = []
    for i in range(Z.shape[1]):
        for j in range(i, Z.shape[1]):
            cols.append(Z[:, i] + Z[:, j]*3571)
    return np.hstack((Z, np.vstack(cols).T))

def create_triples(X):
    cols = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            for k in range(j, X.shape[1]):
                cols.append(X[:, i]*3461 + X[:, j]*5483 + X[:, k])
    return np.hstack((X, np.vstack(cols).T))

testfile=open('test.csv')
testfile.readline()
testdata=[]
id=[]
for line in testfile:
    list=line.rstrip().split(',')
    id.append(int(list[0]))
    temp=[]
    for num in list[1:]:
        temp.append(int(num))
    testdata.append(temp)
testX=np.array(testdata)

modelresult=[]
li=range(70)
count=0
for i in li:
    file=open('train.csv')
    file.readline()
    traindata=[]
    result=[]
    for line in file:
        list=line.rstrip().split(',')
        if int(list[0])==0:
            result.append(int(list[0]))
            temp=[]
            for num in list[1:]:
                temp.append(int(num))
            traindata.append(temp)
        else:
            if random.random()<= 0.1:
                result.append(int(list[0]))
                temp=[]
                for num in list[1:]:
                    temp.append(int(num))
                traindata.append(temp)
    gnb = RandomForestRegressor(30)
    X=np.array(traindata)
    #X=create_triples(X)
    Y=np.array(result)
    print(len(traindata))
    X_,testX_=sparsify(X,testX)

    X_1, X_2, y_1, y_2 = cross_validation.train_test_split(X_, Y, test_size=.20)
    gnb.fit(X_1,y_1)
    
    predictresult=gnb.predict(testX_)

    preds=gnb.predict(X_2)
    fpr, tpr, thresholds = metrics.roc_curve(y_2, preds)
    roc_auc = metrics.auc(fpr, tpr)
    if roc_auc>0.84:
        modelresult.append(predictresult)
        count=count+1

submitfile=open('submission.csv','w')
submitfile.write('Id,Action\n')
i=0
zero=0
one=0

while i<len(id):
    sum=0.0
    for model in modelresult:
        sum=sum+model[i]
    submitfile.write(str(id[i])+","+str(sum/count)+"\n")
    i=i+1

print(count)
