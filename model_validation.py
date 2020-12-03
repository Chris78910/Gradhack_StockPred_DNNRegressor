# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:36:43 2020

@author: 57860
"""

import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GTX 1050 Ti
import matplotlib.pyplot as plt

from keras import models
from keras import layers
Train_s=2214

def error_count(eps,train_data,test_data,train_targets,test_targets):
  model=models.load_model('weight-'+str(eps)+'eps.h5')
  predict_train = model.predict(train_data)
  
  
  predict_test = model.predict(test_data)

  #训练集误差计算
  p_train = predict_train.reshape(1,-1)[0]
  total_train_loss=0
  for i in range(len(p_train)):
    total_train_loss+=abs(p_train[i]-train_targets[i])
  print(len(p_train))
  average_train_loss=total_train_loss/len(p_train)
  print(str(eps)+"次训练集误差"+str(average_train_loss))
  
  #测试集误差计算
  p_test = predict_test.reshape(1,-1)[0]
  
  total_test_loss=0
  for i in range(len(p_test)):
    total_test_loss+=abs(p_test[i]-test_targets[i])
  print(len(p_test))
  average_test_loss=total_test_loss/(len(p_test)-100)
  print(str(eps)+"次测试集误差"+str(average_test_loss))
  
  return p_train

    
data = pd.read_csv("keystats.csv", index_col="Date")
data.dropna(axis=0, how="any", inplace=True)
#print(data.shape)

#feature list f1-f41
features = data.columns[6:].tolist()
#print(features)

#add a column which is the the ratio outperform the s&p500
data['target'] = data["stock_p_change"]/data["Price"]-\
                 data["SP500_p_change"]/data["SP500"]
#index为索引列表                        
data['target']*=100
index = data.columns[6:]
data = data[index]

#print(data.shape)


train_data = data.iloc[0:2500,:-1]
train_targets= data.iloc[0:2500,-1]
#print(train_data)
#test_data = data.iloc[Train_s:2500,:-1]
#test_targets = data.iloc[Train_s:2500,-1]
test_data = data.iloc[Train_s:2500,:-1]
test_targets = data.iloc[Train_s:2500,-1]
#print(test_data)
#print(test_targets)
#data Standardization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


p_train10000 = error_count(10000,train_data,test_data,train_targets,test_targets)
print("------------------------------------------------------")
p_train1000 = error_count(1000,train_data,test_data,train_targets,test_targets)
print("------------------------------------------------------")
p_train100 = error_count(100,train_data,test_data,train_targets,test_targets)
print("------------------------------------------------------")

plt.figure()
plt.ylim(0,200)
plt.plot(range(2500),[abs(p_train10000[i]-train_targets[i]) for i in range(2500)],'b',label='predict-10000')
plt.plot(range(2500),[abs(p_train1000[i]-train_targets[i]) for i in range(2500)],'r',label='predict-1000')
plt.plot(range(2500),[abs(p_train100[i]-train_targets[i]) for i in range(2500)],'g',label='predict-100')
plt.title('Error Statistics-3eps comparison')
plt.legend()


plt.figure()
plt.ylim(0,200)
plt.plot(range(0,2500,10),[abs(p_train10000[i]-train_targets[i]) for i in range(0,2500,10)],'bo',label='predict-10000')
plt.plot(range(0,2500,10),[abs(p_train1000[i]-train_targets[i]) for i in range(0,2500,10)],'ro',label='predict-1000')
plt.plot(range(0,2500,10),[abs(p_train100[i]-train_targets[i]) for i in range(0,2500,10)],'go',label='predict-100')
plt.title('Error Statistics-3eps comparison(scatter)')
plt.legend()

plt.figure()
plt.ylim(0,200)
plt.plot(range(0,2500,10),[abs(p_train10000[i]-train_targets[i]) for i in range(0,2500,10)],'bo',label='predict-10000')
plt.plot(range(0,2500,10),[abs(p_train1000[i]-train_targets[i]) for i in range(0,2500,10)],'ro',label='predict-1000')
plt.title('Error Statistics-2eps comparison(scatter)')
plt.legend()
#error_count(20000,train_data,test_data,train_targets,test_targets)
#error_count(30000,train_data,test_data,train_targets,test_targets)

#drawing comparison

#plt.plot(range(884),[abs(predict10000[i][0]-test_targets[i]) for i in range(884)],'b',label='predict-10000')
#plt.plot(range(884),[x[0] for x in predict50],'g',label='predict-50')
#plt.plot(range(884),test_targets,'r',label='grount-truth')







