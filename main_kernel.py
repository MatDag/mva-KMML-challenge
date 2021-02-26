

from utils import load_data, train_test_split
import numpy as np
#from models import 

import argparse
from pathlib import Path



#from logistic_regression import KernelLogisticRegression
from SVM import SVM
from kernel import *

from sklearn.metrics import accuracy_score



parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="directory where data is stored")
parser.add_argument("--model", required=True, type=str,
                    help="logistic_regression, kernel_logistic_regression, smv")
parser.add_argument("--kernel", required=True, type=str,
                    help="rbf")

parser.add_argument("--gamma", default=0.5, type=float,
                    help="gamma in rbf kernel")




if __name__ == "__main__":
    
    args = parser.parse_args()
    assert args.data_dir.is_dir()


    # load all the data
    X_train_matrix_0,X_train_matrix_1,X_train_matrix_2, X_test_matrix_0, X_test_matrix_1,X_test_matrix_2,Y_train_0,Y_train_1,Y_train_2=load_data()

    X_train_full = np.concatenate((X_train_matrix_0,X_train_matrix_1,X_train_matrix_2))
    Y_train_full = np.concatenate((Y_train_0,Y_train_1,Y_train_2)).reshape(-1)
    X_test_full= np.concatenate((X_test_matrix_0,X_test_matrix_1,X_test_matrix_2))

    Y_train_full[Y_train_full==0]=-1


    X_train, X_val,y_train,y_val = train_test_split(X_train_full,Y_train_full,test_size=0.1)

    print("X_train len :", X_train.shape, "X_val len :" ,X_val.shape,"X_test len :" ,X_test_full.shape )
    # Gram matrix
    

    if args.model=="svm":
        if args.kernel=="linear":
            print("Model svm linear")
            kernel=args.kernel
            Model=SVM()
            params={} # no params
            
            Gram_mat=Linear_Kernel(X_train,**params)
            Gram_mat_val=Linear_Kernel(X_train,X_val,**params)
            print(Gram_mat.shape,y_train.shape)
        
        
            Model.fit(Gram_mat,y_train.reshape(-1,1))
            
        
            y_train_pred = Model.predict_class(Gram_mat_val).reshape(-1)
        
            print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))


            Gram_mat_test=RBF_Kernel(X_train,X_test_full,**params)
            y_pred=Model.predict_class(Gram_mat_test).reshape(-1)


        if args.kernel=="rbf":
            print("Model svm rbf")
            kernel=args.kernel
            Model=SVM()
            params={} # no params
            params['gamma']=args.gamma
            Gram_mat=RBF_Kernel(X_train,**params)
            Gram_mat_val=RBF_Kernel(X_train,X_val,**params)
            print(Gram_mat.shape,y_train.shape)
        
        
            Model.fit(Gram_mat,y_train.reshape(-1,1))
            
        
            y_train_pred = Model.predict_class(Gram_mat_val).reshape(-1)
        
            print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))


            Gram_mat_test=RBF_Kernel(X_train,X_test_full,**params)
            y_pred=Model.predict_class(Gram_mat_test).reshape(-1)

        
        
    if args.model == 'logistic':
        
        if args.kernel=="rbf":
            print("Logistic regression model with kernel {}".format(args.kernel))

            kernel=args.kernel
   
            params={}
            params["gamma"]=args.gamma

            # compute the gram matrix
            Gram_mat=RBF_Kernel(X_train,**params)
            print(Gram_mat)
            
            classifier=KernelLogisticRegression()
            classifier.fit(kernel_train=Gram_mat,label=y_train,lambda_regularisation=0.01)

            
            Gram_mat_val=RBF_Kernel(X_train,X_val,**params)
        
            y_train_pred = classifier.predict_class(Gram_mat_val).reshape(-1)

            print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))


            

        if args.kernel=="linear":
            print("Logistic regression model with kernel {}".format(args.kernel))

            kernel=args.kernel

        
            params={}
            params["gamma"]=args.gamma

            # compute the gram matrix
            Gram_mat=Linear_Kernel(X_train,**params)
            
            classifier=KernelLogisticRegression()
            classifier.fit(kernel_train=Gram_mat,label=y_train,lambda_regularisation=0.01)
            print("Model trained")
            
            Gram_mat_val=Linear_Kernel(X_train,X_val,**params)

        
            y_train_pred = classifier.predict_class(Gram_mat_val).reshape(-1)
            
            print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))
         


    ########################################################## Writte the solution file file !
    with open("submission.csv", 'w') as f:
        f.write(',Bound\n')
        for i in range(len(y_pred)):
            f.write(str(i)+','+str(y_pred[i])+'\n')













