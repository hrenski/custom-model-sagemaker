#! /usr/bin/env python3

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from statsmodels.gam.api import GLMGam, BSplines

class GAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, df = 15, alpha = 1.0, degree = 3):
        self.df = df
        self.alpha = alpha
        self.degree = degree
    
    def fit(self, X, y):
        X, y = self._validate_data(X, y, y_numeric=True)
        
        self.spline = BSplines(
            X, df = [self.df] * self.n_features_in_, 
            degree = [self.degree] * self.n_features_in_, 
            include_intercept = False
        )
        
        gam = GLMGam(
            y, exog = np.ones(X.shape[0]), 
            smoother = self.spline, alpha = self.alpha
        )
        self.gam_predictor = gam.fit()
        
        return self

    def predict(self, X):
        check_is_fitted(self, attributes = "gam_predictor")
        X = check_array(X)
        
        return self.gam_predictor.predict(
            exog = np.ones(X.shape[0]), 
            exog_smooth = X
        )
    
    @property
    def summary(self):
        return self.gam_predictor.summary() if \
               hasattr(self, "gam_predictor") else None
