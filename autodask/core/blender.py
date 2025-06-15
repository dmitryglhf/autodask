import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

from autodask.core.data import ModelContainer
from autodask.utils.log import get_logger
from autodask.utils.regular_functions import is_classification_task, setup_metric, get_n_classes


class WeightedAverageBlender(BaseEstimator):
    """Weighted average ensemble blender.

    Args:
        model_containers (list): List of ModelContainer instances containing trained models.
        task (str): 'classification' or 'regression'.
        max_iter (int, optional): Maximum optimization iterations.

    Attributes:
        weights (np.ndarray): Optimized weights for each model.
    """

    def __init__(
        self,
        model_containers: list,
        task: str,
        max_iter: int = 100
    ):
        self.model_containers = model_containers
        self.task = task
        self.max_iter = max_iter
        self.n_classes = None

        # Result
        self.weights = None

        self.log = get_logger(self.__class__.__name__)
        self.score_func, _, _ = setup_metric(task=task)

    def _fit(self, X, y):
        """Optimize model weights"""
        if len(self.model_containers) == 1:
            self.log.info(f"Got only one model; using weight 1.0 for {self.model_containers[0].model_name}")
            self.weights = np.array([1.0])
            return self

        self.log.info(f"Starting weights optimization for models: {[mc.model_name for mc in self.model_containers]}")

        preds_list = self._get_model_predictions()
        n_models = len(self.model_containers)
        initial_weights = np.ones(n_models) / n_models

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n_models

        def objective(weights):
            blended_pred = self._blend_predictions(preds_list, weights, return_labels=False)
            score = self.score_func(y, blended_pred)
            return score

        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter}
        )
        self.weights = result.x

    def fit(self, X, y):
        """Fit models ensemble"""
        # Preparation
        self.n_classes = get_n_classes(y)
        self._input_validation()

        # Get weights
        self._fit(X, y)

        # Return result
        sorted_pairs = sorted(zip(self.model_containers, self.weights), key=lambda x: x[1], reverse=True)
        weight_formula = " + ".join([f"{round(w, 3)} * {mc.model_name}" for mc, w in sorted_pairs])
        self.log.info(f"Final prediction = {weight_formula}")
        return self

    def predict(self, X):
        """Make predictions using the weighted ensemble"""
        preds_list = []
        for mc in self.model_containers:
            pred = mc.model.predict_proba(X)
            preds_list.append(pred)
        return self._blend_predictions(preds_list, self.weights, return_labels=True)

    def predict_proba(self, X):
        """Predict class probabilities"""
        preds_list = []
        for mc in self.model_containers:
            pred = mc.model.predict_proba(X)
            preds_list.append(pred)
        return self._blend_predictions(preds_list, self.weights, return_labels=False)

    def _blend_predictions(self, predictions: list[np.ndarray], weights: np.ndarray, return_labels: bool = True):
        """Blend predictions using weighted average"""
        blended = np.average(predictions, axis=0, weights=weights)

        if is_classification_task(self.task) and return_labels:
            if self.n_classes is not None:
                if self.n_classes > 2:
                    return np.argmax(blended, axis=1)
                else:
                    return (blended > 0.5).astype(int)
        return blended

    def _get_model_predictions(self):
        """Get oof-predictions from all models"""
        preds_list = []

        for model_container in self.model_containers:
            if hasattr(model_container, 'oof_preds'):
                preds = model_container.oof_preds
                preds_list.append(np.asarray(preds))
            else:
                raise ValueError("Can't obtain oof-predictions from models containers.")

        return preds_list

    def _input_validation(self):
        """Validate input model containers"""
        if not getattr(self, 'model_containers', None):
            raise ValueError(
                "No valid model containers found. "
                "Please provide at least one ModelContainer in model_containers attribute."
            )
        if not all(isinstance(m, ModelContainer) for m in self.model_containers):
            raise TypeError("All elements in model_containers must be ModelContainer instances")

        for model_container in self.model_containers:
            if model_container.model_task_type != self.task:
                raise ValueError(
                    f"Model '{model_container.model_name}' is for task '{model_container.model_task_type}', "
                    f"but blender is for task '{self.task}'"
                )

        if self.task == 'classification' and self.n_classes == 1:
            raise ValueError(
                "For classification tasks, `n_classes` must be > 1 (got n_classes=1)."
            )
