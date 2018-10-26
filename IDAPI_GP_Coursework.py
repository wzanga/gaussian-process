import numpy as np
from scipy.optimize import minimize

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

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 2:
    # TODO: Implement a draw from a multivariate Gaussian here
    
    # if Z follows N(0,I) then  X=A+BZ follows N(A,BB')
    B = np.linalg.cholesky(cov)
    rand_draw = np.random.normal(size=mean.shape[0])
    sample = mean + np.dot(B,rand_draw)
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
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
        for p in range(n):
            for q in range(n):
                tau2 = (np.linalg.norm(X[p]-X[q]))**2
                covMat[p,q] = self.sigma2_f * np.exp(-.5*tau2/(self.length_scale**2)) 

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        #print( "mean_fa",mean_fa.shape)
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        
        #compute useful matrices
        cM_x_a = self.k.covMatrix(self.X,Xa)
        k_x_a = cM_x_a[:self.X.shape[0],self.X.shape[0]:]
        k_a_a = cM_x_a[self.X.shape[0]:,self.X.shape[0]:] - self.k.sigma2_n*np.identity(Xa.shape[0])
        k_x_x = cM_x_a[:self.X.shape[0],:self.X.shape[0]]
        #computational will use the decomposition LL'= K = k(X,X)
        L = np.linalg.cholesky(k_x_x)
        S = np.linalg.solve(L,k_x_a)
        #computation of the mean and covariance
        mean_fa = np.dot(S.T, np.linalg.solve(L, self.y))
        cov_fa = k_a_a - np.dot(S.T,S)
        
        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        y_invK_y = np.dot( self.y.T , np.linalg.solve(self.K,self.y))
        sign, logdet = np.linalg.slogdet(self.K)
        mll = 0.5*( y_invK_y + logdet + self.K.shape[0]*np.log(2*np.pi))
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
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        
        #parameters
        [sf, ls, sn] = np.exp(self.k.getParams())
                        
        #gradient of the covariance matrix k(X,X)
        grad_K_sn = 2*sn*np.identity(self.y.shape[0],float)
        grad_K_sf = np.zeros_like(grad_K_sn)
        grad_K_ls = np.zeros_like(grad_K_sn)
        
        for p in range(self.y.shape[0]):
            for q in range (self.y.shape[0]):
                tau = np.linalg.norm(self.X[p] - self.X[q])
                exp_arg = np.exp(-0.5*(tau/ls)**2)
                grad_K_sf[p][q] = 2*sf*exp_arg
                grad_K_ls[p][q] = ((tau*sf)**2)/(ls**3) * exp_arg
        
        #argument of the trace
        a = np.linalg.solve(self.K , self.y)
        b = np.dot(a, a.T) - np.linalg.inv(self.K)

        #grad_ln
        grad_ln_sigma_f = -0.5 * np.trace( np.dot(b,grad_K_sf) ) *sf
        grad_ln_length_scale = -0.5 * np.trace( np.dot(b,grad_K_ls) ) *ls
        grad_ln_sigma_n = -0.5 * np.trace( np.dot(b,grad_K_sn) )*sn
        
        # Combine gradients
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
        mse = np.mean([ (ya[i]-fbar[i])**2 for i in range(ya.shape[0]) ])
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        sig2 = np.diag(cov) + self.k.sigma2_n
        L=[np.log(2*np.pi*sig2[i])+(ya[i]-fbar[i])**2/sig2[i] for i in range(ya.shape[0])]
        msll = np.mean( 0.5 * np.array(L))
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
    X, y, Xa, ya = loadData('yacht_hydrodynamics.data')
    param = [0.5*np.log(0.1), np.log(0.1), 0.5*np.log(0.5)]
    #RBF and GPR init
    k = RadialBasisFunction(param)
    GPR = GaussianProcessRegression(X,y,k)
    #prediction
    mean_fa,cov_fa=GPR.predict(Xa)
    #gradLML
    grad = GPR.gradLogMarginalLikelihood(param)
    grad_01=[ -35.60076096,   -7.51218753, -165.79906089]
    #opt-param
    #param = GPR.optimize( GPR.k.getParams())
    #mse
    mse=GPR.mse(ya,mean_fa)
    #msll
    msll = GPR.msll(ya,mean_fa,cov_fa)
    