from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

import numpy as np
from sklearn.model_selection import KFold
kf = KFold(10)

def normalizedata(X_train):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    return X_train


def model1(trdata,tract,tsdata):
    model = DecisionTreeClassifier().fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

def model2(trdata,tract,tsdata):
    model = DecisionTreeClassifier(criterion='entropy').fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

def modeln1(trdata,tract,tsdata):
    model = MultinomialNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred
def modeln2(trdata,tract,tsdata):
    model = BernoulliNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred
def modeln3(trdata,tract,tsdata):
    model = GaussianNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

for i in range(1,57):
    print(i)
    fname='data/'+str(i)+'.csv'
    data=np.genfromtxt(fname,delimiter=',')
    data[:,0:-1]=normalizedata(data[:,0:-1])
    in1=np.where(data[:,-1]>0)
    data[in1[0],-1]=1
    predvalue=np.zeros((np.shape(data)[0],6))
    for train_index, test_index in kf.split(data):
        trdata=data[train_index,0:-1]
        tsdata=data[test_index,0:-1]
        tract=data[train_index,-1]
        tsact=data[test_index,-1]
        predvalue[test_index,0]=model1(trdata,tract,tsdata)
        predvalue[test_index,1]=model2(trdata,tract,tsdata)
        predvalue[test_index,2]=modeln1(trdata,tract,tsdata)
        predvalue[test_index,3]=modeln3(trdata,tract,tsdata)
        predvalue[test_index,4]=modeln3(trdata,tract,tsdata)
        predvalue[test_index,5]=tsact
    fname='output/'+str(i)+'.csv'    
    np.savetxt(fname,predvalue, delimiter=',', fmt='%f')  



from sklearn.metrics import (
    f1_score, precision_score, recall_score,accuracy_score
    )



fileloc='output/'
fval=np.zeros((56,5))
acv=np.zeros((56,5))
for i in range(0,56):
    fname= fileloc+str(i+1)+'.csv'
    data1=np.genfromtxt(fname,delimiter=',')
    y1=data1[:,-1]
    for j in range(0,5):
         fval[i,j]=f1_score(y1, data1[:,j])
         acv[i,j]=accuracy_score(y1, data1[:,j])   
fname=fileloc+'acc.csv'    
np.savetxt(fname,acv, delimiter=',', fmt='%f')      
fname=fileloc+'fmea.csv'    
np.savetxt(fname,fval, delimiter=',', fmt='%f') 