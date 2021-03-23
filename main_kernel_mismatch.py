

import sys
sys.path.append('General')
sys.path.append('SVM')

import pandas as pd
import numpy as np
from time import time

from kernel_functions import kernel_spectrum
from spectrum_toolbox import preindexation,Spectrum_embedding, Mismatch_embedding

from SVM import SVM

from sklearn.model_selection import KFold
import scipy.sparse as sp

import matplotlib.pyplot as plt

from time import time



def GridSearch_Mismatch(X,y,hyperparameters,K = 6):
    n_lengths = len(hyperparameters['lengths'])
    n_lambdas = len(hyperparameters['lambdas'])
    n_m = len(hyperparameters['ms'])
    
    y[y==0] = -1
    
    scores_mean = np.zeros((n_lengths,n_m))
    scores_std = np.zeros((n_lengths,n_m))
    
    kf = KFold(n_splits = K)
    
    params = dict()
    
    for i in range(n_lengths):
        print(i)
        for m_count,m in enumerate(hyperparameters['ms']):
            params['k'] = hyperparameters['lengths'][i]
            preindex = preindexation(params['k'])
            X_emb = Mismatch_embedding(X,params['k'],m,preindex = preindex)

            for j in range(n_lambdas):
                c = hyperparameters['lambdas'][j]
                acc = []

                for train_idx,test_idx in kf.split(X):
                    model = SVM(c = c)
                    Xtrain,Xtest = X_emb[train_idx,:],X_emb[test_idx,:]
                    ytrain,ytest = y[train_idx],y[test_idx]              
                    Ktrain = kernel_spectrum(Xtrain,Xtrain,{})
                    Ktest = kernel_spectrum(Xtrain,Xtest,{})
                    model.fit(Ktrain,ytrain[:,None])
                    ypred = model.predict_class(Ktest)

                    acc.append((ypred==ytest).mean())

                scores_mean[i,m_count] = np.array(acc).mean()
                scores_std[i,m_count] = np.array(acc).std()
            del(preindex)
    return(scores_mean,scores_std)



##########################

X = pd.read_csv('data/Xtr0.csv')
y = pd.read_csv('data/ytr0.csv')
X.set_index('Id',inplace = True)

######################################   Grid search  #####################################################""
hyperparameters = dict()
hyperparameters['lengths'] = np.arange(4,5) # for 5,6, 7   ### best c'est 5 et 1
hyperparameters['ms'] = np.arange(2,4) # 3
hyperparameters['lambdas'] = np.array([1e-2])  # tune before 
 #########################################################################################
start = time()

mean,std = GridSearch_Mismatch(X['seq'].to_numpy(),y['Bound'].to_numpy(),hyperparameters)
print('mean',mean,'std',std)


print('time',time()-start)

# for k in range(len(hyperparameters['lengths'])):
#     plt.semilogx(hyperparameters['lambdas'],mean[k,:],label = "$k = {0}$".format(hyperparameters['lengths'][k]))
# plt.legend()





# Xtest = pd.read_csv('data/Xte2.csv')
# Xtest = Spectrum_embedding(Xtest['seq'].to_numpy(),k,preindex = preindex)
# Ktest = kernel_spectrum(Xtrain,Xtest,{})
# ypred2 = model2.predict_class(Ktest)
# ypred2[ypred2==-1]=0

# ypred0 = np.hstack((pd.read_csv('data/Xte0.csv')['Id'].to_numpy()[:,None],ypred0[:,None]))
# ypred1 = np.hstack((pd.read_csv('data/Xte1.csv')['Id'].to_numpy()[:,None],ypred1[:,None]))
# ypred2 = np.hstack((pd.read_csv('data/Xte2.csv')['Id'].to_numpy()[:,None],ypred2[:,None]))

# pd.DataFrame(ypred, columns = ["Id", "Bound"]).set_index('Id').to_csv("SVM_Spectrum.csv"
