# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:08:15 2020

@author: 57860
"""
import pandas as pd
#from tensorflow import keras
from keras import models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def getFundemental(data,tickername):
    data=data[data['Ticker']==tickername]
    #print(data)
    features = data.columns[1:-1]
    X_test = data[features].values
    #print(X_test.shape)
    #print(X_test)
    return X_test

def stock_pred(X_test,eps):
    model=models.load_model('weight-'+str(eps)+'eps.h5')
    pred = model.predict(X_test)
    return pred[0][0]



if __name__ == "__main__":
    data = pd.read_csv("forward_sample3.csv")
    data.dropna(axis=0, how="any", inplace=True)
    stocklist=data['Ticker'].tolist()
    print("there are "+str(len(stocklist))+" Tickers which open to predict")
    print(stocklist)
    
    
    #Here types the ticker name(input) suggest TEL,RTN,TXN,T when demo in case of some weired number caused by abnormal data
    #given prediction represents the exceeding or falling behind rate of a stock while comparing to
    #SP500index at a year after the timing point when fundemental information for a parcific stock is provided
    tickername='TEL'
    X_test=getFundemental(data,tickername)
    print("-----------------------------------------------")
    print("the stock your pick's annual performance against S&P500 index")
    print(str(stock_pred(X_test,10000))+"%")