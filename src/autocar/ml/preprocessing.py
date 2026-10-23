import numpy as np
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self,features):
        self.mean_ = np.mean(features,axis=0)
        self.scale_ = np.std(features,axis=0)
        self.scale_[self.scale_== 0.0] = 1.0
        return self
    def transform(self,features):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("no mean and standard deviation, pls fit() first.")
        return (features-self.mean_)/self.scale_ 
    def fit_transform(self,features):
        self.fit(features)
        return self.transform(features)
    def inverse_transform(self,features):
        return features*self.scale_+self.mean_
