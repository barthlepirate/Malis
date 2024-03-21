import numpy as np


class Perceptron:

    def __init__(self):
        ''' 
        Defines the class two parameters w and b, first as None.
        '''
        self.w = None
        self.b = None

    def train(self,X_train,y_train,lr = 0.01) :
        '''
        Training of the model following the perceptron algorithm.
        Parameters : 
        X_train - Matrix with input features
        y_train - Vector of labels
        lr - Learning rate (default value 0.01)

        '''

        self.w = np.zeros(X_train.shape[1]) #initialize w and b to 0
        self.b = 0

        while True : 
            m=0 #intitialize the misclassification counter
            for i in range(X_train.shape[0]): # iterate through the training dataset
                x = X_train[i,:]
                y = y_train[i]
            
                if (np.dot(self.w.T,x)+self.b)*y <= 0 : # checking if the prediction is incorrect 
                    self.w = self.w + lr*x*y # updating the weights and bias following the stochastic gradient descent (SGB)
                    self.b = self.b + lr*y
                    m = m+1 # update the misclassification counter

            if m == 0: # if no more misclassification are found, break the loop
                break


        return

    
    def predict(self, X):
            '''
            Predicts labels y given an input matrix X
            Parameters: 
            X- matrix of dimensions N x D

            Returns:
            y_pred - vector of labels (dimensions N x 1)
            '''
            
            # Calculate the predictions for each input
            predictions = np.dot(X,self.w) + self.b
            
            # Apply sign function to get binary predictions (-1 or 1)
            y_pred = np.sign(predictions)
            
            return y_pred
    
  