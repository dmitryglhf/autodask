import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy.optimize import minimize

from autodask.utils.log import get_logger
from autodask.utils.regular_functions import is_classification_task, setup_metric, get_n_classes


class WeightedAverageBlender(BaseEstimator):
    """Weighted average ensemble blender.

    Parameters:
        fitted_models (list): List or list of dictionaries containing trained models.
        task (str): 'classification' or 'regression'.
        max_iter (int, optional): Maximum optimization iterations.

    Attributes:
        weights (np.ndarray): Optimized weights for each model.

    Examples:

        >>> # Models must be fitted
        >>> models = [{'model': fitted_model1},
        ...           {'model': fitted_model2, 'name': 'Model2'}]
        >>> # Or
        >>> models = [fitted_model1, fitted_model2]
        >>> blender = WeightedAverageBlender(models, task='classification')
        >>> blender.fit(X_train, y_train)
        >>> preds = blender.predict(X_test)

    """

    def __init__(self,
                 fitted_models: list,
                 task: str,
                 max_iter: int = 100):
        self.fitted_models = fitted_models
        self.task = task
        self.max_iter = max_iter
        self.n_classes = None

        # Result
        self.weights = None

        self.log = get_logger(self.__class__.__name__)
        self.score_func, _, _ = setup_metric(task=task)

    def _fit(self, X, y):
        """Optimize model weights"""
        if len(self.fitted_models) == 1:
            self.log.info(f"Got only one model; using weight 1.0 for {self.fitted_models[0]['name']}")
            self.weights = np.array([1.0])
            return self

        self.log.info(f"Starting weight optimization for models: {[model['name'] for model in self.fitted_models]}")

        preds_list = self._get_model_predictions(X)
        n_models = len(self.fitted_models)
        initial_weights = np.ones(n_models) / n_models

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n_models

        def objective(weights):
            blended_pred = self._blend_predictions(preds_list, weights, return_labels=False)
            score = self.score_func(y, blended_pred)
            return score

        result = minimize(objective, initial_weights,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': self.max_iter})
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
        sorted_pairs = sorted(zip(self.fitted_models, self.weights), key=lambda x: x[1], reverse=True)
        weight_formula = " + ".join([f"{round(w, 3)} * {model['name']}" for model, w in sorted_pairs])
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
        preds_list = [model['model'].predict_proba(X) for model in self.fitted_models]
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
        else:
            return blended

    def _get_model_predictions(self, X):
        """Get predictions from all models"""
        preds_list = []
        for model in self.fitted_models:
            est = model['model']
            if is_classification_task(self.task):
                preds = est.predict_proba(X)
            else:
                preds = est.predict(X)
            preds_list.append(np.asarray(preds))
        return preds_list

    def _input_validation(self):
        processed_models = []

        for item in self.fitted_models:
            if hasattr(item, '__class__') and hasattr(item.__class__, '__name__'):
                model_dict = {
                    'model': item,
                    'name': item.__class__.__name__
                }
                processed_models.append(model_dict)
            elif isinstance(item, dict):
                if 'model' not in item:
                    raise ValueError("Model dictionary must contain 'model' key")

                model = item['model']
                if not (hasattr(model, '__class__') and hasattr(model.__class__, '__name__')):
                    raise ValueError(f"Invalid model object: {model}")

                model_dict = item.copy()
                if 'name' not in model_dict or not str(model_dict.get('name', '')).strip():
                    model_dict['name'] = model.__class__.__name__
                processed_models.append(model_dict)

            else:
                raise ValueError(f"Invalid item in models list: {item}. Must be either a model or a dictionary")

        self.fitted_models = processed_models

    def _models_validation(self):
        for model_dict in self.fitted_models:
            model = model_dict['model']
            model_name = model_dict['name']
            try:
                check_is_fitted(model)
            except NotFittedError:
                raise ValueError(
                    f"Model '{model_name}' is not fitted. "
                    "All ensemble models must be fitted."
                )

        if self.task == 'classification' and self.n_classes == 1:
            raise ValueError(
                "For classification tasks, `n_classes` must be > 1 (got n_classes=1)."
            )
