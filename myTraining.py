# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:13:51 2020

@author: Diwakar
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
def data_split(data,ratio):
    #np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
if __name__=="__main__":
    df=pd.read_csv('data1.csv')
    train,test=data_split(df,0.3)
    X_train = train[['feaver','bodyPain','age','runnyNose','diffBreath']].values
    X_test = test[['feaver','bodyPain','age','runnyNose','diffBreath']].values
    Y_train = train[['infectionProb']].values.reshape(777,)
    Y_test = test[['infectionProb']].values.reshape(333,)
    clf=LogisticRegression()
    clf.fit(X_train,Y_train)
    #Open a file,where you want to store the data
    file=open('model.pk','wb')
    #dump information to the file
    pickle.dump(clf,file)
    file.close()
    