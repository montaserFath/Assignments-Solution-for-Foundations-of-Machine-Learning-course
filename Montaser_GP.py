

import numpy as np
import os
from scipy.optimize import minimize
from numpy import linalg as LA
from numpy.linalg import inv
# we use the following for plotting figures in jupyter
#get_ipython().magic('matplotlib inline')

# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]
    return X_train, y_train, X_test, y_test
                                ########Q1###################
    
def multivariateGaussianDraw(mean, cov):
    sample = np.random.multivariate_normal(mean,cov) # This is only a placeholder
    # Task 2:
    # TODO: Implement a draw from a multivariate Gaussian here

    # Return drawn sample
    return sample


                        ##################Q 2,3,4#######################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)
        
    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug
        
        n = X.shape[0]
        covMat = np.zeros((n,n))
        
        # Task 1:
        # TODO: Implement the covariance matrix here
        
#         for i in range (n):
#             for j in range (n):
#                 diff=-1*(1/2)*((LA.norm(X[i]-X[j]))**2)*(self.length_scale)
#                 covMat[i][j]=(self.sigma2_f)* np.exp(diff)

        sq=np.sum(X**2,1).reshape(-1,1)+np.sum(X**2,1) -2*np.dot(X,X.T)
        covMat=self.sigma2_f * np.exp(-0.5 / self.length_scale * sq)
        
        
        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat
                            ############Q 5,6,7#######################
        
class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        N=Xa.shape[0]-self.n #Xtest.shape[0]
        k_all=self.KMat(Xa)
        
#         kx_x=k_all[ :((self.n)+1), :((self.n)+1)]     #shape= nxn
        
#         kx_xtrain=k_all[ :((self.n)+1), :N+1]         #shape= nxN
        
#         kxtrain_x=k_all[:N , :((self.n)+1)]           #shape= Nxn
        
#         kxtrain_xtrain=k_all[:N+1 , :N+1]             #shape= NXN

        Kxx=self.KMat(self.X)
        Kss=self.KMat(Xa)
        Kxs=self.k.covMatrix(self.X,Xa)[:231,231:]
        #Kxs=self.KMat(Xa,self.X)[0:231,231:308]
        Kxs = Kxs.T
        
        mean_fa= Kxs @ inv(Kxx) @ self.y                        #shape=Nx1
        cov_fa= Kss - (Kxs  @ inv(Kxx) @ Kxs.T)  #shape=NXN
        # Return the mean and covariance
   
        return mean_fa, cov_fa
    
    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        Kxx=self.KMat(self.X)
        
        
        mll = 0.5 *((self.y.T @ inv (Kxx) @ self.y)+ np.log(np.linalg.det(Kxx))+ self.n * np.log(2*np.pi))
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y

        # Return mll
        return mll
    
    
    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    
    
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        
        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
         # Combine gradients
            
        matrix=((inv(K)@ self.y) @ (inv(K)@ self.y).T)- inv(K)
        I=np.eye(K.shape[0])
        I-I* self.k.ln_sigma_n**2
        d=K-I
        df=(2*self.k.ln_sigma_f**2)*(K-I)

        grad_ln_sigma_n=0.5*np.trace(matrix*(2*self.k.ln_sigma_n**2))
        
                
                
        grad_ln_sigma_f=0.5*np.trace(matrix @ df)
        
        
        aa=np.zeros((self.n, self.n))
        for i in range (0,(self.n)):
            for j in range (0, (self.n)):
                aa[i][j]=(LA.norm(self.X[i]- self.X[j])**2)/(self.k.length_scale**2)
                
        grad_ln_length_scale=0.5* np.trace(matrix @ d @ aa)
        
        
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])
        # Return the gradients
        return gradients

    
    
 
        
    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        #mse=ya.T @ ya + fbar.T @ fbar -2*ya @ fbar
        summ=0
        for i in range (len(ya)):
            summ+=(ya[i]-fbar[i])**2
        mse=summ/len(ya)
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        cov=cov.diagonal()+self.k.sigma2_n
        summ=0
        for i in range (len(ya)):
            summ+=((ya[i]-fbar[i])**2)/(cov[i])+np.log(2*np.pi*cov[i])
        msll=summ/(2*len(ya))
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x
    
    
    
if __name__ == '__main__':

    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################



