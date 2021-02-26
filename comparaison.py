# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:14:42 2021

@author: pierre

Test on a Sklearn example to see if SMV works
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from kernel import RBF_Kernel,Linear_Kernel



from SVM import SVM

X, y = make_classification(n_samples=100, random_state=0)
y[y==0]=-1
X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=0)
clf = svm.SVC(kernel='precomputed')
# linear kernel computation
gram_train = np.dot(X_train, X_train.T)
clf.fit(gram_train, y_train)

# predict on training examples
gram_test = np.dot(X_test, X_train.T)  # in sklearn dimention of gram matrix are nb_composante, nb_sample
pred=clf.predict(gram_test)




#################################################################### Our code
#

Gram_train=Linear_Kernel(X_train)
Gram_test=Linear_Kernel(X_train,X_test)

Model=SVM()
classif=Model.fit(Gram_train,y_train.reshape(-1,1))  # need y_train with shape (n,1)
predi=Model.predict_class(Gram_test)  # Need Gram_test with shape nb_samples,nb_composante)



sklearn_accuracy=accuracy_score(y_test,pred)
our_accuracy=accuracy_score(y_test,predi)

print('Linear classifier :skearn accuracy :',sklearn_accuracy,"our_accuracy",our_accuracy)

########################################### Test with rbf kernel

params={}
params['gamma']= 0.1
Gram_train=RBF_Kernel(X_train,**params)
Gram_test=RBF_Kernel(X_train,X_test,**params)

Model=SVM()
classif=Model.fit(Gram_train,y_train.reshape(-1,1))  # need y_train with shape (n,1)
predi=Model.predict_class(Gram_test)  # Need Gram_test with shape nb_samples,nb_composante)



sklearn_accuracy=accuracy_score(y_test,pred)
our_accuracy=accuracy_score(y_test,predi)

clf2=svm.SVC(gamma='auto')  # put to 0.1 to get the same result than our
clf2.fit(X_train, y_train)


pred=clf2.predict(X_test)
print(pred,predi)

sklearn_accuracy=accuracy_score(y_test,pred)

our_accuracy=accuracy_score(y_test,predi)

print("RBF classifuer," ",sklearn_accuracy,", sklearn_accuracy ,",our_accuracy)", our_accuracy)











