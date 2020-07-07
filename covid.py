# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:24:11 2020

@author: Diwakar
"""

import pandas as pd
import numpy as np
#To read csv file
df=pd.read_csv('data1.csv')
def data_split(data,ratio):
    #np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
train,test=data_split(df,0.3)
X_train = train[['feaver','bodyPain','age','runnyNose','diffBreath']].values
X_test = test[['feaver','bodyPain','age','runnyNose','diffBreath']].values
Y_train = train[['infectionProb']].values.reshape(777,)
Y_test = test[['infectionProb']].values.reshape(333,)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0)
clf.fit(X_train,Y_train)
infProb=clf.predict_proba([[98,1,22,-1,1]])[0][1]

