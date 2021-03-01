# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:40:36 2021

@author: matth
"""

import numpy as np 
from kernel_functions import kernels_dic

default_lambda = 1e-5

#from sklearn.kernel_ridge import KernelRidge

#%%

# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", reg = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += reg * np.identity(k_matrix.shape[0])
    
    # The test matrix 
    t_matrix = kernel(X_test, X_train, param)
    
    # Regression coefficients in feature space
    coeff = np.linalg.solve(k_matrix, Y_train) #np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    prediction = np.matmul(t_matrix, coeff)
    
    return prediction, coeff

def kernel_regression_coeff(X_train, Y_train, param, kernel_keyword = "RBF", reg = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += reg * np.identity(k_matrix.shape[0])
    
    # Regression coefficients in feature space
    coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    return coeff


#%%
class KernelRegression():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
    def fit(self, X_train, Y_train, reg = default_lambda):
        self.coeff = kernel_regression_coeff(X_train, Y_train, self.parameters, kernel_keyword = self.kernel_keyword, reg = reg)
        self.X_train = np.copy(X_train)
        
    def predict(self, X_test, reg = default_lambda):  
        kernel = kernels_dic[self.kernel_keyword]
        test_matrix = kernel(X_test, self.X_train, self.parameters)
        pred = np.dot(test_matrix, self.coeff)

        return pred
    
    def fit_predict(self, X_train, X_test, Y_train, reg = default_lambda):
        self.fit(X_train, Y_train, reg = reg)
        pred = self.predict(X_test)
        
        return pred

#%%
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.random((100,2))
    Y = np.random.random((100,1))
    
    X_train = X[:80]
    Y_train = Y[:80]
    
    X_test = X[80:]
    Y_test = Y[80:]
    
    mu = np.array([0.5])
    pred, coeff = kernel_regression(X_train, X_test, Y_train, mu)
    
    KR = KernelRegression("RBF", mu)
    KR.fit(X_train, Y_train)
    pred_2 = KR.predict(X_test)
    
    """
    gamma = 1/mu**2
    K = KernelRidge(alpha = default_lambda, kernel = "rbf", gamma = gamma)
    K.fit(X_train, Y_train)
    
    pred_3 = K.predict(X_test)
    """