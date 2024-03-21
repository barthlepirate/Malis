import numpy as np

class RidgeReg():
  def __init__(self, alpha=1.0):
      self.alpha = alpha
    
  def fit(self, X, y):
    X_augmented = np.c_[np.ones((X.shape[0], 1)), X]  # Augment X with a column of ones
    A = np.identity(X_augmented.shape[1])
    A[0, 0] = 0

    A_biased = self.alpha * A
    self.thetas = np.linalg.inv(X_augmented.T.dot(X_augmented) + A_biased).dot(X_augmented.T).dot(y)
    return self


  def predict(self, X):
      X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
      self.predictions = X_predictor.dot(self.thetas)
      return self.predictions