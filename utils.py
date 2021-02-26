
import pandas as pd
import numpy as np
import random
####read data
import os
#from Classifiers.KernelSVM import SVMC

def load_data():
    list_dir = os.listdir('.')
    if 'data' in list_dir:
        root = '.'
    else:
        root='..'
    '''
    Read all the datasets
    '''
    X_train_0 = (pd.read_csv(root+ r'/data/Xtr0.csv',header=None).values).tolist()
    X_train_1 = (pd.read_csv(root+r'/data/Xtr1.csv',header=None).values).tolist()
    X_train_2 = (pd.read_csv(root+r'/data/Xtr2.csv',header=None).values).tolist()
    
    X_train_matrix_0 = (pd.read_csv(root+r'/data/Xtr0_mat100.csv',sep=' ',header=None).values)
    X_train_matrix_1 = (pd.read_csv(root+r'/data/Xtr1_mat100.csv',sep=' ',header=None).values)
    X_train_matrix_2 = (pd.read_csv(root+r'/data/Xtr2_mat100.csv',sep=' ',header=None).values)
    
    Y_train_0 = (pd.read_csv(root+r'/data/Ytr0.csv',sep=',',index_col=0).values)
    Y_train_1 = (pd.read_csv(root+r'/data/Ytr1.csv',sep=',',index_col=0).values)
    Y_train_2 = (pd.read_csv(root+r'/data/Ytr2.csv',sep=',',index_col=0).values)
    
    X_test_0 = (pd.read_csv(root+r'/data/Xte0.csv',header=None).values).tolist()
    X_test_1 = (pd.read_csv(root+r'/data/Xte1.csv',header=None).values).tolist()
    X_test_2 = (pd.read_csv(root+r'/data/Xte2.csv',header=None).values).tolist()
    
    X_test_matrix_0 = (pd.read_csv(root+r'/data/Xte0_mat100.csv',sep=' ',header=None).values)
    X_test_matrix_1 = (pd.read_csv(root+r'/data/Xte1_mat100.csv',sep=' ',header=None).values)
    X_test_matrix_2 = (pd.read_csv(root+r'/data/Xte2_mat100.csv',sep=' ',header=None).values)
    
    
    X_train_0 = (np.array(X_train_0)[:,0]).tolist()
    X_train_1 = np.array(X_train_1)[:,0].tolist()
    X_train_2 = np.array(X_train_2)[:,0].tolist()
    
    X_test_0 = (np.array(X_test_0)[:,0]).tolist()
    X_test_1 = np.array(X_test_1)[:,0].tolist()
    X_test_2 = np.array(X_test_2)[:,0].tolist()
    return X_train_matrix_0,X_train_matrix_1,X_train_matrix_2, X_test_matrix_0, X_test_matrix_1, X_test_matrix_2,Y_train_0,Y_train_1,Y_train_2



def train_test_split(*arrays,test_size=0.2):
    '''
    Function that split arrays and list in two sets 
    Param: *arrays: arrays to split
    test_size: (float) size of the test dataset (between 0 and 1)
    '''
    list_to_return =  []
    shape_data = len(arrays[0])
    list_indice_shuffle= list(range(shape_data))
    random.shuffle(list_indice_shuffle)
    list_train, list_test = list_indice_shuffle[:int(len(list_indice_shuffle)*(1-test_size))],list_indice_shuffle[int(len(list_indice_shuffle)*(1-test_size)):]
    for array in arrays:
        if isinstance(array,list):
            list_to_return.extend([list(np.array(array)[list_train]),list(np.array(array)[list_test])])
        else:
            list_to_return.extend([(np.array(array)[list_train]),(np.array(array)[list_test])])
    return list_to_return





