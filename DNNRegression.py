# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:46:18 2020

@author: 57860
"""

import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GTX 1050 Ti

from keras import models
from keras import layers
Train_s=2500
eps=10000

data = pd.read_csv("keystats.csv", index_col="Date")
data.dropna(axis=0, how="any", inplace=True)
print(data.shape)

#feature list f1-f41
features = data.columns[6:].tolist()
print(features)

#add a column which is the the ratio outperform the s&p500
data['target'] = data["stock_p_change"]/data["Price"]-\
                 data["SP500_p_change"]/data["SP500"]
#index为索引列表                        
data['target']*=100
index = data.columns[6:]
data = data[index]
#print(data)
#print(data.shape)


train_data = data.iloc[0:Train_s,:-1]
train_targets= data.iloc[0:Train_s,-1]
test_data = data.iloc[2500:,:-1]
test_targets = data.iloc[2500:,-1]

#data Standardization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


import numpy as np
from keras import backend as K
K.clear_session()
import matplotlib.pyplot as plt

model = build_model()
history=model.fit(train_data, train_targets, epochs=eps, batch_size=50, verbose=1)

#drawing learning curve
epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['loss'],'b',label='loss')
plt.title("learning_curve")
plt.legend()

#save model, learning_curve
plt.savefig("learning_curve-"+str(eps)+"eps.jpg")
model.save("weight-"+str(eps)+"eps.h5")