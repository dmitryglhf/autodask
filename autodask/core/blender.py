import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.optimize import minimize_scalar

from utils.log import get_logger
from utils.regular_functions import is_classification_task, setup_metric


class WeightedAverageBlender(BaseEstimator):
    def __init__(self,
                 fitted_models: list,
                 task: str,
                 metric: str = None,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 n_classes: int = None):
        """
        Ensemble Blender for combining predictions of fitted models.

        Parameters:
        - fitted_models: list of dicts, each dict with keys:
            'model': a trained model supporting predict / predict_proba
            'name': str, name of the model
        - task: str, either 'classification' or 'regression'
        - metric: str or callable, evaluation metric
        - max_iter: int, max number of coordinate descent iterations
        - tol: float, tolerance for early stopping
        - n_classes: int, required for classification
        """
        self.fitted_models = fitted_models
        self.task = task
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.n_classes = n_classes
        self.weights = None

        self.log = get_logger(self.__class__.__name__)
        self.score_func, self.metric_name, self.is_maximize_metric = setup_metric(metric_name=metric, task=task)

    def fit(self, X, y):
        """
        Fit ensemble weights using coordinate descent.

        Parameters:
        - X: array-like or DataFrame, input features for inference
        - y: array-like, true labels or targets
        """
        if len(self.fitted_models) == 1:
            self.weights = np.array([1.0])
            self.log.info(f"Only one model; using weight 1.0 for {self.fitted_models[0]['name']}")
            return self

        self.log.info(f"Starting weight optimization for models: {[model['name'] for model in self.fitted_models]}")

        # Precompute model predictions
        preds_list = self._get_model_predictions(X)
        preds_array = pd.concat(preds_list, axis=1).values
        n_models = len(self.fitted_models)
        self.weights = np.ones(n_models) / n_models

        for _ in range(self.max_iter):
            old_weights = self.weights.copy()

            for i in range(n_models):
                def objective(wi):
                    w = self.weights.copy()
                    w[i] = wi
                    w /= w.sum()
                    y_pred = self._predict_weighted(w, preds_array)
                    score = self.score_func(y, y_pred)
                    return -score if self.is_maximize_metric else score

                res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
                self.weights[i] = res.x

            self.weights /= self.weights.sum()

            if np.max(np.abs(self.weights - old_weights)) < self.tol:
                break

        model_weights = {model['name']: round(w, 6) for model, w in zip(self.fitted_models, self.weights)}
        self.log.info(f"Obtained weight-model pairs: {model_weights}")
        return self

    def predict(self, X):
        """
        Make predictions with the blended model.

        Parameters:
        - X: input data

        Returns:
        - array-like: predicted labels or values
        """
        preds_list = self._get_model_predictions(X)
        preds_array = pd.concat(preds_list, axis=1).values
        return self._predict_weighted(self.weights, preds_array)

    def predict_proba(self, X):
        """
        Predict probabilities for classification tasks.

        Parameters:
        - X: input data

        Returns:
        - np.ndarray: predicted class probabilities
        """
        if not is_classification_task(self.task):
            raise ValueError(f"predict_proba is only available for classification tasks.")
        preds_list = [model['model'].predict_proba(X) for model in self.fitted_models]
        preds_array = np.concatenate(preds_list, axis=1)
        _, probs = self._apply_weights_classification(self.weights, preds_array)
        return probs

    def _predict_weighted(self, weights: np.ndarray, features: np.ndarray):
        if is_classification_task(self.task):
            labels, _ = self._apply_weights_classification(weights, features)
            return labels
        else:
            return np.dot(features, weights)

    def _apply_weights_classification(self, weights: np.ndarray, features: np.ndarray):
        n_samples = features.shape[0]
        probs = np.zeros((n_samples, self.n_classes))

        for class_idx in range(self.n_classes):
            class_preds = np.zeros((n_samples, len(weights)))
            for model_idx in range(len(weights)):
                col_idx = model_idx * self.n_classes + class_idx
                class_preds[:, model_idx] = features[:, col_idx]
            probs[:, class_idx] = np.sum(class_preds * weights, axis=1)

        labels = np.argmax(probs, axis=1)
        return labels, probs

    def _get_model_predictions(self, X):
        """
        Compute predictions or probabilities from all models.

        Returns:
        - list of DataFrames
        """
        preds_list = []
        for model in self.fitted_models:
            mdl = model['model']
            if is_classification_task(self.task):
                preds = mdl.predict_proba(X)
                preds_list.append(pd.DataFrame(preds))
            else:
                preds = mdl.predict(X)
                preds_list.append(pd.DataFrame(preds if preds.ndim == 2 else preds[:, np.newaxis]))
        return preds_list
