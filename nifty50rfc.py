import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)
data=pd.read_csv('^NSEI.xslx')
data.dropna(axis=0, how='any')
data=data.drop(['Date'],axis=1)
# print(data.head())
data.loc[:,'signal']= data.loc[:,'Close']<data.loc[:,' Future_Close']
# print(data.head())
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
for i in range(0,5):
    data_train=data_train.loc[data_train[i]!='null',:]
    data_test=data_test.loc[data_test[i]!='null',:]
data_train=data_train.astype(float)
data_test=data_test.astype(float)
# scaler=MinMaxScaler()
# scaler.fit(data_train)
# data_train=scaler.transform(data_train)
# data_test=scaler.transform(data_test)
x_train=data_train.iloc[:,0:4]
y_train=data_train.iloc[:, 5]
x_test=data_test.iloc[:,0:4]
y_test=data_test.iloc[:, 5]
# print(x_train.head())
# print(y_train.head())
features = 4

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
pred=clf.predict(x_test)
pred=pred.reshape(489,1)
error=np.sum(np.subtract(pred,(y_test.values.reshape(len(pred),1)))!=0)
print(float(error)/489*100)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# line1, =ax1.plot(y_test, linewidth=0.5)
# line2, = ax1.plot(pred, linewidth=0.5)
# plt.savefig('rfc.jpeg')
pred2=clf.predict(pd.DataFrame([10207,10243,10175,10231]).T)
print(pred2)

#vini vici chakra