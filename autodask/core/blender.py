import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

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
        if len(self.fitted_models) == 1:
            self.weights = np.array([1.0])
            self.log.info(f"Only one model; using weight 1.0 for {self.fitted_models[0]['name']}")
            return self

        self.log.info(f"Starting weight optimization for models: {[model['name'] for model in self.fitted_models]}")

        preds_list = self._get_model_predictions(X)
        n_models = len(self.fitted_models)
        initial_weights = np.ones(n_models) / n_models

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n_models

        def objective(weights):
            return_labels = self.metric_name in ['accuracy', 'f1', 'precision', 'recall']
            blended_pred = self._blend_predictions(preds_list, weights, return_labels=return_labels)
            score = self.score_func(y, blended_pred)
            return -score if self.is_maximize_metric else score

        result = minimize(objective, initial_weights,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': self.max_iter, 'tol': self.tol})
        self.weights = result.x

        model_weights = {model['name']: round(w, 6) for model, w in zip(self.fitted_models, self.weights)}
        self.log.info(f"Obtained weight-model pairs: {model_weights}")
        return self

    def predict(self, X):
        preds_list = self._get_model_predictions(X)
        return self._blend_predictions(preds_list, self.weights, return_labels=True)

    def predict_proba(self, X):
        if not is_classification_task(self.task):
            raise ValueError("predict_proba is only available for classification tasks.")
        preds_list = [model['model'].predict_proba(X) for model in self.fitted_models]
        return np.average(np.stack(preds_list), axis=0, weights=self.weights)

    def _blend_predictions(self, predictions: list[np.ndarray], weights: np.ndarray, return_labels: bool = True):
        """
        Blends predictions using weighted average.

        Parameters:
        - predictions: list of np.ndarray
        - weights: np.ndarray of weights
        - return_labels: if True, return argmax for classification

        Returns:
        - np.ndarray: blended predictions or probabilities
        """
        blended = np.average(np.stack(predictions), axis=0, weights=weights)

        if is_classification_task(self.task):
            if self.n_classes is not None and self.n_classes > 2:
                return np.argmax(blended, axis=1) if return_labels else blended
            else:
                # Binary classification case - treat like regression
                return (blended > 0.5).astype(int) if return_labels else blended
        else:
            return blended

    def _get_model_predictions(self, X):
        preds_list = []
        for model in self.fitted_models:
            est = model['model']
            if is_classification_task(self.task):
                preds = est.predict_proba(X)
            else:
                preds = est.predict(X)
            preds_list.append(np.asarray(preds))
        return preds_list
