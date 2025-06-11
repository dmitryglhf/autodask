import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
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

    Examples:
        >>> # Models must be fitted
        >>> model1 = ModelContainer(fitted_model1, "Model1", "classification")
        >>> model2 = ModelContainer(fitted_model2, "Model2", "classification")
        >>> blender = WeightedAverageBlender([model1, model2], task='classification')
        >>> blender.fit(X_train, y_train)
        >>> preds = blender.predict(X_test)
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

        self.log.info(f"Starting weight optimization for models: {[mc.model_name for mc in self.model_containers]}")

        preds_list = self._get_model_predictions(X)
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
        self._input_validation()
        self._models_validation()
        self.n_classes = get_n_classes(y)

        # Get weights
        self._fit(X, y)

        # Return result
        sorted_pairs = sorted(zip(self.model_containers, self.weights), key=lambda x: x[1], reverse=True)
        weight_formula = " + ".join([f"{round(w, 3)} * {mc.model_name}" for mc, w in sorted_pairs])
        self.log.info(f"Final prediction = {weight_formula}")
        return self

    def predict(self, X):
        """Make predictions using the weighted ensemble"""
        preds_list = self._get_model_predictions(X)
        return self._blend_predictions(preds_list, self.weights, return_labels=True)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if not is_classification_task(self.task):
            raise ValueError("predict_proba is only available for classification tasks.")
        preds_list = [mc.predict_proba(X) for mc in self.model_containers]
        return np.average(np.stack(preds_list), axis=0, weights=self.weights)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def fit_predict_proba(self, X, y):
        return self.fit(X, y).predict_proba(X)

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

    def _get_model_predictions(self, X):
        """Get predictions from all models with method availability checks"""
        preds_list = []

        for model_container in self.model_containers:
            try:
                if is_classification_task(self.task):
                    preds = model_container.predict_proba(X)
                else:
                    preds = model_container.predict(X)

                preds_list.append(np.asarray(preds))

            except Exception as e:
                raise RuntimeError(
                    f"Error getting predictions from model {model_container.model_name}: {str(e)}"
                ) from e

        return preds_list

    def _input_validation(self):
        """Validate input model containers"""
        if not isinstance(self.model_containers, list) or len(self.model_containers) == 0:
            raise ValueError("model_containers must be a non-empty list of ModelContainer instances")

        for mc in self.model_containers:
            if not isinstance(mc, ModelContainer):
                raise ValueError(f"All items in model_containers must be ModelContainer instances, got {type(mc)}")

    def _models_validation(self):
        """Validate that all models are fitted and compatible with the task"""
        for model_container in self.model_containers:
            try:
                check_is_fitted(model_container.model)
            except NotFittedError:
                raise ValueError(
                    f"Model '{model_container.model_name}' is not fitted. "
                    "All ensemble models must be fitted."
                )

            if model_container.model_task_type != self.task:
                raise ValueError(
                    f"Model '{model_container.model_name}' is for task '{model_container.model_task_type}', "
                    f"but blender is for task '{self.task}'"
                )

        if self.task == 'classification' and self.n_classes == 1:
            raise ValueError(
                "For classification tasks, `n_classes` must be > 1 (got n_classes=1)."
            )
