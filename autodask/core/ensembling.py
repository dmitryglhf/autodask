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

    def fit(self, X, y):
        """Weights optimization"""
        self.log.info(f"Starting weights optimization for models {...}")

        n_models = len(self.fitted_models)
        weights = np.ones(n_models) / n_models

        for _ in range(self.max_iter):
            old_weights = weights.copy()

            for i in range(n_models):
                # Фиксируем все веса, кроме i-го
                def objective(wi):
                    w = weights.copy()
                    w[i] = wi
                    w /= w.sum()
                    # ensemble_pred = sum(w[j] * models_val_preds[j] for j in range(n_models))

                    # return self.score_func(y, ensemble_pred)

                # Weight i-го optimization
                res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
                weights[i] = res.x

            weights /= weights.sum()

            # Early stopping
            if np.max(np.abs(weights - old_weights)) < self.tol:
                break

        return weights

    def predict(self, X):
        """Return predictions as weighted average"""
        if not self.fitted_models:
            raise ValueError("No models in ensemble")

        # здесь каждая из моделей должны сделать предскзание на X_test
        # и затем нужно умножить их на веса, полученные в fit
