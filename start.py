#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:55:38 2021

@author: mathieudagreou
"""

"""
We reproduce in this script, our prediction using a kernel SVM with a spectrum 
kernel.
"""

import numpy as np
import pandas as pd

import sys
sys.path.append('General')
sys.path.append('SVM')

from kernel_functions import kernel_spectrum
from spectrum_toolbox import preindexation,Spectrum_embedding
from SVM import SVM

# Loading the dataset

Xtrain = [pd.read_csv('data/Xtr{0}.csv'.format(k)) for k in range(3)]
y = [pd.read_csv('data/ytr{0}.csv'.format(k)) for k in range(3)]
Xtest  = [pd.read_csv('data/Xte{0}.csv'.format(k)) for k in range(3)]

ytest = []

# Parameters for each datasets
lengthes_subseq = [8,6,7]
tab_c = [1e-2, 1e-3, 1e-2]

for k in range(3):
    preindex = preindexation(lengthes_subseq[k]) # Compute a preindexation dictionnary
    Xtrain_emb = Spectrum_embedding(Xtrain[k]['seq'], lengthes_subseq[k],preindex = preindex) # Compute the spectrum embedding
    Ktrain = kernel_spectrum(Xtrain_emb,Xtrain_emb,{}) # Compute the kernel
    
    ytrain = y[k]['Bound'].to_numpy()[:,None]
    ytrain[ytrain==0]=-1
    
    model = SVM(c=tab_c[k])
    model.fit(Ktrain,ytrain) # Fit the model
    
    Xtest_emb = Spectrum_embedding(Xtest[k]['seq'], lengthes_subseq[k],preindex = preindex)
    Ktest = kernel_spectrum(Xtrain_emb,Xtest_emb,{}) # Compute the test kernel
    
    ypred = model.predict_class(Ktest) # Prediction
    ypred[ypred==-1]=0
    ytest.append(np.hstack((Xtest[k]['Id'].to_numpy()[:,None],ypred[:,None])))

y = np.vstack(ytest)

pd.DataFrame(y, columns = ["Id", "Bound"]).set_index('Id').to_csv("Yte.csv")