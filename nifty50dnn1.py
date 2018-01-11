import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import keras
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1)
cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Future_Close']
data = pd.read_csv('^NSEI_withoutnull.csv', header=0, names=cols)
data = data.drop(['Date'], axis=1)
arr = data.copy()
arr = arr.dropna(axis=0, how='any')
train_start=0
train_end=int(np.floor(0.8*arr.shape[0]))
test_start=train_end+1
test_end=int(arr.shape[0])
arr = arr.values
# shuffle_indices = np.random.permutation(np.arange(2466))
# arr=arr[shuffle_indices]
data_train=arr[np.arange(train_start, train_end),:]
data_test=arr[np.arange(test_start,test_end),:]
data_train=pd.DataFrame(data_train)
data_test=pd.DataFrame(data_test)
data_train=data_train.astype(float)
data_test=data_test.astype(float)
data_train.columns = cols[1:] 
data_test.columns = cols[1:]
data_train['Close'] = pd.to_numeric(data_train['Close'], errors='coerce').fillna(0).astype(float)
data_train['Future_Close'] = pd.to_numeric(data_train['Future_Close'], errors='coerce').fillna(0).astype(float)
data_train['Ratio'] = data_train['Future_Close']/data_train['Close']
data_test['Close'] = pd.to_numeric(data_test['Close'], errors='coerce').fillna(0).astype(float)
data_test['Future_Close'] = pd.to_numeric(data_test['Future_Close'], errors='coerce').fillna(0).astype(float)
data_test['Ratio'] = data_test['Future_Close']/data_test['Close']
data_train['Direction'] = np.where(data_train['Future_Close'] > data_train['Close'], 1, 0)
data_test['Direction'] = np.where(data_test['Future_Close'] > data_test['Close'], 1, 0)
# scaler=MinMaxScaler()
# scaler.fit(data_train)
# data_train=scaler.transform(data_train)
# data_test=scaler.transform(data_test)
x_train=data_train.iloc[:,0:4]
y_train=data_train.iloc[:, 6]
x_test=data_test.iloc[:,0:4]
y_test=data_test.iloc[:, 6]
features = 4

"""
Logistic Regression
log = LogisticRegression()
log.fit(x_train, y_train)
pred = log.predict(x_test)

Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

Decision Tree Classifier
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)
pred = dec.predict(x_test)

Naïve Bayes Classifier Algorithm
gau = GaussianNB()
gau.fit(x_train, y_train)
pred = gau.predict(x_test)

Support Vector Machines
model = svm.SVC(kernel='sigmoid', C = 100, gamma = 0.001)
model.fit(x_train, y_train)
pred = model.predict(x_test)
"""

pred = pred.reshape(len(pred),1)
error = np.sum(np.subtract(pred,(y_test.values.reshape(len(pred),1)))!=0)
print(float(error)/len(pred)*100)
print(pred)

"""
Errors
RandomForestClassifier = 54.24430641821946
Naïve Bayes Classifier Algorithm = 53.83022774327122
Decision Tree = 53.41614906832298
Logistic Regression = 49.68944099378882
Support Vector Machine = 46.16977225672878
"""
