import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize_scalar

from utils.log import get_logger
from utils.regular_conditions import is_classification_task, setup_metric


class EnsembleBlender(BaseEstimator):
    def __init__(self,
                 fitted_models: list,
                 task: str,
                 metric:str=None,
                 max_iter=100,
                 tol=1e-6):
        self.fitted_models = fitted_models
        self.task = task
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

        self.log = get_logger(self.__class__.__name__)

        # Setup metric function
        self.score_func, self.metric_name, self.maximize_metric = (
            setup_metric(metric_name=metric, task=task)
        )

    def fit(self, y):
        """Weights optimization"""
        self.log.info(f"Starting weights optimization for models {...}")

        if is_classification_task(self.task):
            weighted_pred = self._get_classification_score
        else:
            weighted_pred = self._get_regression_score

        n_models = len(self.fitted_models)
        weights = np.ones(n_models) / n_models

        for _ in range(self.max_iter):
            old_weights = weights.copy()

            for i in range(n_models):
                # Fix all the weights except w_i
                def objective(wi):
                    w = weights.copy()
                    w[i] = wi
                    w /= w.sum()
                    y_pred = weighted_pred(w, y)
                    return self.score_func(y, y_pred)

                # Coordinate descent optimization for w_i
                res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
                weights[i] = res.x

            weights /= weights.sum()

            # Early stopping
            if np.max(np.abs(weights - old_weights)) < self.tol:
                break

        self.log.info(f"Weights-models pairs: {...}")
        return weights

    def predict(self, X):
        """Return predictions as weighted average"""
        if not self.fitted_models:
            raise ValueError("No models in ensemble")

        # здесь каждая из моделей должны сделать предскзание на X_test
        # и затем нужно умножить их на веса, полученные в fit

    def _get_classification_score(self, weights: np.ndarray, y_pred):
        probs = np.zeros((n_samples, n_classes))
        for class_idx in range(n_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                col_idx = model_idx * n_classes + class_idx
                class_preds[:, model_idx] = features[:, col_idx]
            probs[:, class_idx] = np.sum(class_preds * weights, axis=1)

        labels = np.argmax(probs, axis=1)

        # If `target` is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if y_pred is not None:
            score = self.score_func(y_pred, labels)
        else:
            return labels, probs

        return score

    def _get_regression_score(self, weights: np.ndarray, y_true):
        # Get predictions by applying the weights
        predictions = np.dot(features, weights)

        # If `target` is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if y_true is not None:
            score = -self.score_func(y_true, predictions) # minimizing operation for regression
        else:
            return predictions

        return score
