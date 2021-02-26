
import numpy as np


def kernel_rbf(x, y,gamma):
   
    return np.exp(- gamma * np.linalg.norm(x- y)**2)

def linear_kernel(x,y):
    return np.dot(x,y)

### to compute the Gram matrix for any kernel
def Gram(X,kernel,Y=[],**params):
    
    if len(Y)==0:
        len_X = X.shape[0]
        gram_matrix = np.zeros((len_X, len_X), dtype=np.float32)

        for i in range(len_X):
            for j in range(i,len_X):
                gram_matrix[i,j] = kernel(X[i],X[j],**params)
                gram_matrix[j,i] = gram_matrix[i,j]
        
        return gram_matrix
    else:
        len_X = X.shape[0]
        len_Y = Y.shape[0]
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)
        
        for i in range(len_X):
            for j in range(len_Y):
                gram_matrix[i,j] = kernel(X[i],Y[j],**params)
                
        return gram_matrix

###############" functions that return gram matrix with appropriate kernel"
def RBF_Kernel(X,Y=[],**params):
    if len(Y)==0:

        return Gram(X,kernel=kernel_rbf,Y=[],**params)
    else :
        return Gram(X,kernel_rbf,Y,**params)

def Linear_Kernel(X,Y=[],**params):
    if len(Y)==0:

        return Gram(X,kernel=linear_kernel,Y=[],**params)
    else :
        return Gram(X,kernel=linear_kernel,Y=Y,**params)
    



                
        
