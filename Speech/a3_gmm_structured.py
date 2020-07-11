import numpy as np
import os, fnmatch
import random
from contextlib import redirect_stdout

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        return ((self._d/2)*np.log(2*np.pi)) + \
            np.sum(np.square(self.mu[m])/(2*self.Sigma[m])) + \
            0.5*np.sum(np.log(self.Sigma[m]))

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    return - myTheta.precomputedForM(m) - np.sum(
            (0.5*np.square(x)*(1/myTheta.Sigma[m])) -  
            (myTheta.mu[m]*x*(1/myTheta.Sigma[m])), axis=-1) # txd * dx1 * 1xd = txd
    
def stable_logsumexp(array_like, axis=-1):
    """Compute the stable logsumexp of `vector_like` along `axis`
    This `axis` is used primarily for vectorized calculations.
    """
    array = np.asarray(array_like)
    # keepdims should be True to allow for broadcasting
    m = np.max(array, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(array - m), axis=axis))

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    return (np.log(myTheta.omega) + log_Bs) - stable_logsumexp(np.log(myTheta.omega) + log_Bs, axis = 0)
 
def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    return np.sum(stable_logsumexp(np.add(np.log(myTheta.omega),log_Bs), axis = 0))

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 28)
     # initialize mu to M random rows of X. 
    X_preserved = X.copy()
    np.random.shuffle(X)
    myTheta.reset_mu(X[:M])
    X = X_preserved.copy()
    myTheta.reset_omega(np.random.dirichlet(np.ones(M),size=1)) 
    myTheta.reset_Sigma(np.ones((myTheta.Sigma.shape[0], myTheta.Sigma.shape[1])))
    #loop
    i = 0
    prev_L = float(-np.inf)
    improvement = float(np.inf)
    while i <= maxIter and improvement >= epsilon:
        # compute intermediate results
        log_Bs = np.zeros((M, X.shape[0]))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, X, myTheta)
        log_Ps = log_p_m_x(log_Bs, myTheta) # M
        # compute log likelihood
        L = logLik(log_Bs, myTheta)
        # update parameters
        myTheta.reset_omega(np.sum(np.exp(log_Ps), axis=1)/log_Ps.shape[1])
        # numerator = for t in range(T):  c += np.outer(a[:,t],b[t, :])
        interm = np.sum(np.einsum('mt,td->tmd', np.exp(log_Ps), X), axis=0)
        myTheta.reset_mu(interm / np.sum(np.exp(log_Ps), axis=1, keepdims=True)) 
        interm = np.sum(np.einsum('mt,td->tmd', np.exp(log_Ps), np.square(X)), axis=0)
        myTheta.reset_Sigma(interm / np.sum(np.exp(log_Ps), axis=1, keepdims=True) - np.square(myTheta.mu))
        improvement = L - prev_L
        prev_L = L
        i += 1
    return myTheta

def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    # mfcc is a t x d vector
        # model - is object theta     
    bestModel = -1
    log_likes = []
    for i in range(len(models)):  
        M = models[i]._M
        # get logBs
        log_Bs = np.zeros((M, mfcc.shape[0]))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, mfcc, models[i])
        # append log likelihood for log_likes 
        log_likes.append(logLik(log_Bs, models[i]))
    # use arg sort to get top K 
    best_K = np.argsort(log_likes)[-k:]
    log_likes = np.array(log_likes)
    models = np.array(models)
    best_Ls = log_likes[best_K]
    best_models = models[best_K]
    # update best model
    bestModel = best_K[-1]
    # write output into a file
    print(str(correctID))
    for i in range(k-1, -1,-1):
        print(f'{best_models[i].name} {best_Ls[i]}')

    return 1 if (bestModel == correctID) else 0

if __name__ == "__main__":
    with open('gmmDiscussion.txt', 'w') as f, redirect_stdout(f):
            trainThetas = []
            testMFCCs = []
            d = 13
            k = 5   # number of top speakers to display, <= 0 if none
            M = 8
            epsilon = 0.0
            maxIter = 20
            # train a model for each speaker, and reserve data for testing

            for subdir, dirs, files in os.walk(dataDir):
                for speaker in dirs:
                    print(speaker)

                    files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
                    random.shuffle(files)

                    testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                    testMFCCs.append(testMFCC)

                    X = np.empty((0, d))

                    for file in files:
                        myMFCC = np.load(os.path.join(dataDir, speaker, file))
                        X = np.append(X, myMFCC, axis=0)

                    trainThetas.append(train(speaker, X, M, epsilon, maxIter))

            # evaluate
            numCorrect = 0

            for i in range(0, len(testMFCCs)):
                numCorrect += test(testMFCCs[i], i, trainThetas, k)
            accuracy = 1.0 * numCorrect / len(testMFCCs)
