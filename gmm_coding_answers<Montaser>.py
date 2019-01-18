import numpy as np
from scipy import stats
Multi_N= stats.multivariate_normal
class GaussianMixtureModel():
    """Density estimation with Gaussian Mixture Models (GMM).

    You can add new functions if you find it useful, but *do not* change
    the names or argument lists of the functions provided.
    """
    def __init__(self, X, K):
        """Initialise GMM class.

        Arguments:
          X -- data, N x D array
          K -- number of mixture components, int
        """
        self.X = X
        self.n = X.shape[0]
        self.D = X.shape[1]
        self.K = K

    def MNormal(self, mu, S, X):
        result = np.zeros((self.n,self.K))
        for i in range(self.n):
            for j in range(self.K):
                result[i,j] = Multi_N.pdf(self.X[i],mu[j],S[j], allow_singular=True)
        return result
    def mixure(self, mu_k,S_k, pi_k,x):
        result = []
        for i in range(self.K):
            #print(i, self.normal(x, mu_k[i], S_k[i]))
            result.append(Multi_N.pdf(x, mu_k[i], S_k[i], allow_singular=True))
        return (pi_k * np.array(result).reshape(-1,1)).sum()
            
    def log_likelihood(self,mu,S, pi, X):
        l = 0
        n = X.shape[0]
        for i in range(n):
            #print(i)
            #if i in [27,28,29]:
                #print(self.mixure(mu,S, pi,self.X[i]))
            l += np.log(self.mixure(mu,S, pi, X[i]))
            
        return l
    
    def E_step(self, mu, S, pi):
        """Compute the E step of the EM algorithm.

        Arguments:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array

        Returns:
          r_new -- updated component responsabilities, N x K array
        """
        #print(mu.shape == (self.K, self.D))
        #print(S.shape)
        #print(S.shape  == (self.K, self.D, self.D))
        #print(pi.shape == (self.K, 1))
        # Assert that all arguments have the right shape
        assert(mu.shape == (self.K, self.D) and               S.shape  == (self.K, self.D, self.D) and               pi.shape == (self.K, 1))
        r_new = np.zeros((self.n, self.K))

        # Task 1: implement the E step and return updated responsabilities
        # Write your code from here...
        #r_new = pi[i]*Multi_N.pdf(self.X[i],mu[1],S[i])
        r_new = pi.reshape(1,-1)*self.MNormal(mu, S, self.X)
        r_new /= r_new.sum(axis = 1).reshape(-1,1)
        # ... to here.
        assert(r_new.shape == (self.n, self.K))
        return r_new


    def M_step(self, mu, r):
        """Compute the M step of the EM algorithm.

        Arguments:
          mu -- previous component means, K x D array
          r -- previous component responsabilities,  N x K array

        Returns:
          mu_new -- updated component means, K x D array
          S_new -- updated component covariances, K x D x D array
          pi_new -- updated component weights, K x 1 array
        """
        assert(mu.shape == (self.K, self.D) and               r.shape  == (self.n, self.K))
        mu_new = np.zeros((self.K, self.D))
        S_new  = np.zeros((self.K, self.D, self.D))
        pi_new = np.zeros((self.K, 1))

        # Task 2: implement the M step and return updated mixture parameters
        # Write your code from here...
        #mu_new = np.sum(r[:,i]*self.X, axis=1)/r[:,i].sum()
        Nk = np.sum(r, axis=0).reshape(-1,1)
        
        mu_new = (r.T @ self.X)/Nk
        
        #S_new = np.sum(r[:,i]*(self.X[i] - mu[i]).T @ (self.X[i] - mu[i]), axis=1)/r[:,i].sum() 
        for i in range(self.K):
            S_new[i] = r[:,i]*(self.X - mu_new[i]).T @ (self.X - mu_new[i])/r[:,i].sum() 
        
        pi_new = Nk/self.n
        # ... to here.
        assert(mu_new.shape == (self.K, self.D) and               S_new.shape  == (self.K, self.D, self.D) and               pi_new.shape == (self.K, 1))
        return mu_new, S_new, pi_new

    
    def train(self, initial_params):
        """Fit a Gaussian Mixture Model (GMM) to the data in matrix X.

        Arguments:
          initial_params -- dictionary with fields 'mu', 'S', 'pi' and 'K'

        Returns:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array
          r -- component responsabilities, N x K array
        """
        # Assert that initial_params has all the necessary fields
        assert(all([k in initial_params for k in ['mu', 'S', 'pi']]))

        mu = np.zeros((self.K, self.D))
        S  = np.zeros((self.K, self.D, self.D))
        pi = np.zeros((self.K, 1))
        r  = np.zeros((self.n, self.K))

        # Task 3: implement the EM loop to train the GMM
        # Write your code from here....
        mui=initial_params['mu']
        Si=initial_params['S']
        pii=initial_params['pi']
        r = self.E_step(mui, Si, pii)
        l_old = 0
        i = 0
        niter=1
        for j in range(niter):#while np.abs(self.log_likelihood(mu, S, pi, self.X) - l_old) > 1e-30:#
            if j==0:
                r = self.E_step(mui, Si, pii)
                mu, S, pi = self.M_step(mu, r)
                i +=1
            if i%100 == 0:
                #print('Step number: ' ,i)
                r = self.E_step(mu, S, pi)
                mu, S, pi = self.M_step(mu, r)
                i +=1

        # ... to here.
        assert(mu.shape == (self.K, self.D) and               S.shape  == (self.K, self.D, self.D) and               pi.shape == (self.K, 1) and               r.shape  == (self.n, self.K))
        #print ('log=',self.log_likelihood(mu,S, pi, self.X))
        return mu, S, pi, r
