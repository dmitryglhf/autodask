import numpy as np
from sklearn.base import BaseEstimator


class EnsembleBlender(BaseEstimator):
    def __init__(self, fitted_models: list, maximize_metric=True):
        self.maximize_metric = maximize_metric
        self.fitted_models = fitted_models

    def fit(self, X, y):
        """Weights optimization"""
        return self

    def predict(self, X=None):
        """Return predictions as weighted average"""
        if not self.fitted_models:
            raise ValueError("No models in ensemble")
