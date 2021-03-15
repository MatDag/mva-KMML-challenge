# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:41:51 2021

@author: matth
"""

import cvxpy as cp
import numpy as np
import sys
sys.path.append('../General')
from kernel_functions import kernels_dic

default_lambda = 1e-5
#%%

# The logistic loss (assumes y = +-1)
def logistic_loss(weights, Y, K, reg = 1e-5):
    norm = cp.quad_form(weights, K)
    n = weights.shape[0]
    f_x = K@weights 

    return norm*reg/2 + cp.sum(cp.logistic(f_x) - cp.multiply(f_x, Y), axis = 0)/n

def solve_kernel_LR(X, Y,  parameters, kernel_keyword = "RBF",reg = default_lambda):
    kernel = kernels_dic[kernel_keyword]
    
    n = X.shape[0]
    K = kernel(X, X, parameters)
    
    weights = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(logistic_loss(weights, Y, K, reg)))
    problem.solve()
    return np.array(weights.value)

def logistic(weights, K):
    return 1/(1+ np.exp(-K@weights))

def logistic_prediction(weights, X_train, X_test, parameters, kernel_keyword = "RBF"):
    kernel = kernels_dic[kernel_keyword]
    K = kernel(X_test, X_train, parameters)
    
    prob = logistic(weights, K)
    
    prob[prob > 0.5] = 1
    prob[prob <= 0.5] = 0
    
    return prob
    
#%%

class KernelLogisticRegression():
    def __init__(self, kernel_keyword, parameters, custom_kernel = None):
        self.kernel_keyword = kernel_keyword
        
        # For custom kernels
        if self.kernel_keyword == "custom":
            kernels_dic["custom"] = custom_kernel
        self.parameters = np.copy(parameters)
    def fit(self, X_train, Y_train, reg = default_lambda):
        weights_opt = solve_kernel_LR(X_train, Y_train,  self.parameters, 
                                      kernel_keyword = self.kernel_keyword,reg =reg)
        
        self.weights = weights_opt
        self.X_train = X_train.copy()
        
    def predict(self, X_test):
        pred = logistic_prediction(self.weights, self.X_train, X_test, self.parameters, kernel_keyword = self.kernel_keyword)
        
        return pred
#%%

if __name__ == "__main__":
    mu = np.array([1.0])
    n = 10
    X = np.random.random((n, 5))
    Y = np.random.binomial(1, 0.5, size = n)
    X_test = np.random.random((8, 5))
    
    weights_opt = solve_kernel_LR(X, Y,  mu, kernel_keyword = "RBF",reg = default_lambda)
    
    print(weights_opt)
    
    print(Y)
    print(logistic_prediction(weights_opt, X, X, mu))
    print(logistic_prediction(weights_opt, X, X_test, mu))
    
    # Checking the class implementation
    
    K = KernelLogisticRegression("RBF", mu)
    K.fit(X, Y)
    
    print(K.weights)
    print(K.predict(X_test))