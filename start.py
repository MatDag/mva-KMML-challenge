#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:55:38 2021

@author: mathieudagreou
"""

import numpy as np
import pandas as pd

import sys
sys.path.append('General')
sys.path.append('SVM')

from kernel_functions import kernel_spectrum
from spectrum_toolbox import preindexation,Spectrum_embedding

from SVM import SVM

Xtrain = [pd.read_csv('data/Xtr{0}.csv'.format(k)) for k in range(3)]
y = [pd.read_csv('data/ytr{0}.csv'.format(k)) for k in range(3)]

Xtest  = [pd.read_csv('data/Xte{0}.csv'.format(k)) for k in range(3)]

ytest = []

lengthes_subseq = [8,6,7]
tab_c = [1e-2, 1e-3, 1e-2]

for k in range(3):
    preindex = preindexation(lengthes_subseq[k])
    Xtrain_emb = Spectrum_embedding(Xtrain[k]['seq'], lengthes_subseq[k],preindex = preindex)
    Ktrain = kernel_spectrum(Xtrain_emb,Xtrain_emb,{})
    
    ytrain = y[k]['Bound'].to_numpy()[:,None]
    ytrain[ytrain==0]=-1
    
    model = SVM(c=tab_c[k])
    model.fit(Ktrain,ytrain)
    
    Xtest_emb = Spectrum_embedding(Xtest[k]['seq'], lengthes_subseq[k],preindex = preindex)
    Ktest = kernel_spectrum(Xtrain_emb,Xtest_emb,{})
    
    ypred = model.predict_class(Ktest)
    ypred[ypred==-1]=0
    ytest.append(np.hstack((Xtest[k]['Id'].to_numpy()[:,None],ypred[:,None])))

y = np.vstack(ytest)

pd.DataFrame(y, columns = ["Id", "Bound"]).set_index('Id').to_csv("Yte.csv")