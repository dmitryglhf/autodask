import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.optimize import minimize_scalar

from utils.log import get_logger
from utils.regular_functions import is_classification_task, setup_metric


class EnsembleBlender(BaseEstimator):
    def __init__(self,
                 fitted_models: list,
                 task: str,
                 metric:str=None,
                 max_iter=100,
                 tol=1e-6,
                 n_classes=None):
        self.fitted_models = fitted_models
        self.task = task
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

        self.log = get_logger(self.__class__.__name__)

        # Setup metric function
        self.score_func, self.metric_name, self.is_maximize_metric = (
            setup_metric(metric_name=metric, task=task)
        )
        self.features = self._get_joined_features()
        self.n_models = len(fitted_models)
        self.n_samples = self._get_num_samples()
        self.n_classes = n_classes

    def fit(self, X, y):
        """Weights optimization"""
        if self.n_models == 1:
            self.weights = 1.0
            self.log.info(f"Weights: {self.weights}*{self.fitted_models[0]['model'][1]}")
            return self

        self.log.info(f"Starting weights optimization for models:"
                      f" {[model['model'][1] for model in self.fitted_models]}")

        if is_classification_task(self.task):
            weighted_pred = self.apply_weights_classification
        else:
            weighted_pred = self.apply_weights_regression

        n_models = len(self.fitted_models)
        weights = np.ones(n_models) / n_models

        for _ in range(self.max_iter):
            old_weights = weights.copy()

            for i in range(n_models):
                # Lock all the weights except w_i
                def objective(wi):
                    w = weights.copy()
                    w[i] = wi
                    w /= w.sum()
                    y_pred = weighted_pred(w, self.features, y)
                    return self.score_func(y, y_pred)

                # Coordinate descent optimization for w_i
                res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
                weights[i] = res.x

            weights /= weights.sum()

            # Early stopping
            if np.max(np.abs(weights - old_weights)) < self.tol:
                break

        model_weight_dict = dict(zip([model['model'] for model in self.fitted_models], weights.round(6)))
        self.log.info(f"Obtained weight-model pairs: {model_weight_dict}")
        self.weights = weights
        return self

    def predict(self, X):
        """Return predictions as weighted average"""
        if not self.fitted_models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models and join it to the array by columns
        if is_classification_task(self.task):
            preds_list = [model['model'][0].predict_proba(X) for model in self.fitted_models]
            preds_array = pd.concat(preds_list, axis=1)
            y_pred, _ = self.apply_weights_classification(self.weights, preds_array)
        else:
            preds_list = [model['model'][0].predict(X) for model in self.fitted_models]
            preds_array = pd.concat(preds_list, axis=1)
            y_pred = self.apply_weights_regression(self.weights, preds_array)

        return y_pred

    def predict_proba(self, X):
        """Return probabilities as weighted average"""
        if not self.fitted_models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models and join it to the array by columns
        if is_classification_task(self.task):
            preds_list = [model['model'][0].predict_proba(X) for model in self.fitted_models]
            preds_array = pd.concat(preds_list, axis=1)
            _, probs = self.apply_weights_classification(self.weights, preds_array)
        else:
            raise ValueError(f"This method is unavailable for task {self.task}")

        return probs

    def apply_weights_classification(self, weights: np.ndarray, features, target=None):
        probs = np.zeros((self.n_samples, self.n_classes))
        for class_idx in range(self.n_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((self.n_samples, self.n_models))
            for model_idx in range(self.n_models):
                col_idx = model_idx * self.n_classes + class_idx
                class_preds[:, model_idx] = features[:, col_idx]
            probs[:, class_idx] = np.sum(class_preds * weights, axis=1)

        labels = np.argmax(probs, axis=1)

        # If 'target' is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if target is not None:
            return self.score_func(target, labels)
        else:
            return labels, probs

    def apply_weights_regression(self, weights: np.ndarray, features, target=None):
        # Get predictions by applying the weights
        predictions = np.dot(features, weights)

        # If 'target' is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if target is not None:
            return -self.score_func(target, predictions) # minimizing operation for regression
        else:
            return predictions

    def _get_joined_features(self):
        preds_list = [model['preds'] for model in self.fitted_models]
        return pd.concat(preds_list, axis=1)

    def _get_num_samples(self):
        return len(self.fitted_models[0]['preds'])
