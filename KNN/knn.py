from scipy.spatial import distance_matrix
from scipy import stats
import numpy as np

class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        ''' 

        self.X=X.copy()  
        self.y=y.copy()       
       
    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 

        dst = self.minkowski_dist(X_new, p) 
        ordered = np.argsort(dst, axis=1) # Orders all distances in ascending order
        neighbors = self.y[ordered[:,0:self.k]] #For every point in the test set, picks the k closest points in the training set
        y_hat, _ = stats.mode(neighbors, axis=1) # Assigns labels to the new data

        return y_hat
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        
        dst = distance_matrix(X_new, self.X, p) 
        return dst
    


