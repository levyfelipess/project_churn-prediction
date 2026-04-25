import numpy as np
from scipy.stats import uniform

class BinaryRandomGuess:
    def __init__(self, threshold=0.5, seed=None):
        self.threshold = threshold
        self._uniform = uniform(0, 1)
        self._seed = seed

    def fit(self, X, y):
        pass

    def predict(self, X):
        n = X.shape[0]
        y_pred = self._uniform.rvs(size=n, random_state=self._seed)
        y_pred = np.int32(y_pred >= self.threshold)
        return y_pred

    def predict_proba(self, X):
        n = X.shape[0]
        y_prob_pred = self._uniform.rvs(size=n, random_state=self._seed).reshape(-1, 1)
        y_prob_pred = np.concatenate((1. - y_prob_pred, y_prob_pred), axis=1)
        return y_prob_pred
