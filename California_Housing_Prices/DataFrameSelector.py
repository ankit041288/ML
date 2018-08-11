from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self, attribute):
        self.attribute=attribute
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return x[self.attribute].values