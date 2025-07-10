from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

class OutliersDetector(BaseEstimator , TransformerMixin):
    def __init__(self , factor=1.5 , activate =True):
        self.factor=factor
        self.activate = activate

    def fit(self,X,y=None):
        q1 = np.nanquantile(X , q = 0.25 , axis=0)
        q3 = np.nanquantile(X , q=0.75 , axis=0)
        iqr = q3-q1
        self.lim_sup_ = q3 + self.factor * iqr
        self.lim_inf_ = q1 - self.factor * iqr
        return self
    def transform(self , X,y=None):
        X_transf = X.copy()
        if self.activate:
            f = (X_transf > self.lim_sup_)|(X_transf<self.lim_inf_)
            X_transf[f] = np.nan
        return pd.DataFrame(X_transf, columns=X.columns, index=X.index) if isinstance(X, pd.DataFrame) else X_transf
    def get_feature_names_out(self, input_features=None):
        return input_features
    

class SqrtTransformer(BaseEstimator , TransformerMixin):
    def __init__(self,activate=True):
        self.activate=activate
    def fit(self,X,y=None):
        return self
    def transform(self , X,y=None):
        X_transf = X.copy()
        if self.activate:
            X_transf = np.sqrt(X_transf)
        return X_transf
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class LogTransformer(BaseEstimator , TransformerMixin):
    def __init__(self,activate=True):
        self.activate= activate
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_transf=X.copy()
        if self.activate:
            X_transf=np.log(X_transf)
        return X_transf
    def get_feature_names_out(self, input_features=None):
        return input_features
    

class DeteccionOutliers(BaseEstimator , TransformerMixin):
    def __init__(self,factor=1.5):
        self.factor=factor
        
    def fit(self,X,y=None):
        q1=np.quantile(X,0.25 , axis=0)
        q3=np.quantile(X,0.75 , axis=0)
        iqr = q3-q1
        self.lim_sup_=q3+self
    
class CategTransformer(BaseEstimator , TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_transf=X.copy()
        for c in X_transf.columns:

            X_transf[c]=X_transf[c].str.findall([r"[0-9]+"]).apply(lambda x: x[0] if isinstance(x,list) and len(x)>0 else np.nan)
        return X_transf
    def get_feature_names_out(self, input_features=None):
        return input_features
    

        

