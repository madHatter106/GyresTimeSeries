import numpy as np

class Scaler:
    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inv_transform(self, X_s):
        return X_s * self.std + self.mean