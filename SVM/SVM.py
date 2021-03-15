
import numpy as np 
import cvxopt
import cvxopt.solvers



class SVM:

    def __init__(self, c=1,min_sv = 1e-4,print_report = False):
        self.alpha_ = None # the solution 
        self.min_sv = min_sv # minimal value to be  a support vector
        self.c = c #regularization
        self.print_report = print_report
        
            
    def fit(self,kernel_train,label):
        '''
        Solve  the  following quadratic optimization : 
            min 1/2 u^T P u + q^T u
            s.t.  Au=b
                  Gu <=h 
        '''
        n = label.shape[0] # number of points

        # Create P and q vectors for cvxopt
        diag = np.zeros((n,n))
        np.fill_diagonal(diag, label)
        q = cvxopt.matrix(np.ones(n) * -1)
        P = np.dot(diag, np.dot(kernel_train, diag))
        P = cvxopt.matrix(P)
        
        
        # Regularization
        if self.c is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G_inf = np.diag(np.ones(n) * -1)
            G_sup = np.identity(n)
            G = cvxopt.matrix(np.vstack((G_inf, G_sup)))
            h_inf = np.zeros(n)
            h_sup = np.ones(n) *self.c
            h = cvxopt.matrix(np.hstack((h_inf, h_sup)))

        # Create A and B 
        A = label.transpose()
        A=A.astype(np.float)
        A = cvxopt.matrix(A, (1, n), 'd')
        b = cvxopt.matrix(0.0)
        
        # Solve optimization prob
        if not(self.print_report):
            cvxopt.solvers.options['show_progress'] = False
        u = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        #  solution of the dual problem
        alpha = np.ravel(u['x'])
        
    
        sv = alpha > self.min_sv # find support vectors
        ind = np.arange(len(alpha))[sv]
        
        self.alpha_ = alpha[sv]
        self.sv = np.argwhere(sv==True)
        self.sv_label = label[sv]
        
        if self.print_report:
            print ("%d support vectors out of %d points" % (len(self.alpha_), n))

        # Bias value/intercept
        self.b = 0*1.0 # initialize b as float
        #compute b
        for i in range(len(self.alpha_)):
            self.b += self.sv_label[i]
            self.b -= np.sum(self.alpha_ * self.sv_label[:,0] * kernel_train[sv,ind[i]])
        self.b /= len(self.alpha_)
        
        
    def get_coef(self):
        
        return list(self.alpha_)

    def predict(self,kernel_test):
        """
        predict prob of be in a class, kernel_test of size np.array(nb_samples,nb_composante))
        """
       
        y_predict = np.zeros(kernel_test.shape[1])
        
        for i in range(kernel_test.shape[1]):
            #print(self.alpha_.shape,self.sv.shape,self.sv,self.sv_label[:,0])
            y_predict[i] = sum(alpha * sv_label * kernel_test[sv,i] for alpha, sv, sv_label in zip(self.alpha_, self.sv, self.sv_label[:,0]))
        return y_predict + self.b

        prediction= np.sign(y_predict + self.b)
        
        return prediction
    
    def predict_class(self,kernel_test):
        '''
        return -1 or +1 
        Paramètres:  (np.array(nb_samples,nb_composante)) 
        
        '''
        prediction = np.array(self.predict(kernel_test)>=0,dtype=int)
        prediction[prediction==0]=-1
    
        return prediction
    


    