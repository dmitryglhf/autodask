import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

from autodask.utils.log import get_logger
from autodask.utils.regular_functions import is_classification_task, setup_metric


class WeightedAverageBlender(BaseEstimator):
    """Weighted ensemble blender that optimizes model weights for best performance.

    This class implements a weighted average ensemble where weights are optimized
    to maximize/minimize the specified metric. Supports both classification and
    regression tasks.

    Parameters:
        fitted_models (list): List of dictionaries containing trained models.
            Each dict should have:
            - 'model': Trained model implementing predict/predict_proba
            - 'name': String identifier for the model
        task (str): Type of task, either 'classification' or 'regression'
        metric (str, optional): Evaluation metric to optimize.
            If None, uses default metric for the task type.
        max_iter (int, optional): Maximum optimization iterations. Defaults to 100.
        tol (float, optional): Tolerance for early stopping. Defaults to 1e-6.
        n_classes (int, optional): Number of classes (for classification).
            Required for multiclass classification. Defaults to None.

    Attributes:
        weights (np.ndarray): Optimized weights for each model
        score_func (callable): Metric scoring function
        metric_name (str): Name of the evaluation metric
        is_maximize_metric (bool): Whether higher metric values are better

    Examples:
        >>> models = [{'model': RandomForestClassifier(), 'name': 'RF'},
        ...           {'model': LogisticRegression(), 'name': 'LR'}]
        >>> blender = WeightedAverageBlender(models, task='classification')
        >>> blender.fit(X_train, y_train)
        >>> preds = blender.predict(X_test)
    """

    def __init__(self,
                 fitted_models: list,
                 task: str,
                 metric: str = None,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 n_classes: int = None):
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
        """Optimize model weights based on validation performance.

        Args:
            X: Feature matrix for weight optimization
            y: Target values for weight optimization

        Returns:
            self: Returns the instance itself

        Notes:
            - Uses constrained optimization to find weights summing to 1
            - For single model, automatically uses weight 1.0
            - Logs final optimized weights for each model
        """
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
        """Make predictions using the weighted ensemble.

        Args:
            X: Feature matrix to predict on

        Returns:
            np.ndarray: Predicted values
                - Class labels for classification
                - Continuous values for regression
        """
        preds_list = self._get_model_predictions(X)
        return self._blend_predictions(preds_list, self.weights, return_labels=True)

    def predict_proba(self, X):
        """Predict class probabilities (classification only).

        Args:
            X: Feature matrix to predict on

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes)

        Raises:
            ValueError: If called for regression tasks
        """
        if not is_classification_task(self.task):
            raise ValueError("predict_proba is only available for classification tasks.")
        preds_list = [model['model'].predict_proba(X) for model in self.fitted_models]
        return np.average(np.stack(preds_list), axis=0, weights=self.weights)

    def _blend_predictions(self, predictions: list[np.ndarray], weights: np.ndarray, return_labels: bool = True):
        """Blend predictions using weighted average.

        Args:
            predictions: List of model predictions
            weights: Array of model weights
            return_labels: Whether to return class labels (True)
                or probabilities/scores (False)

        Returns:
            np.ndarray: Blended predictions
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
        """Get predictions from all models.

        Args:
            X: Feature matrix for predictions

        Returns:
            list: List of numpy arrays containing predictions
                - Probabilities for classification
                - Raw predictions for regression
        """
        preds_list = []
        for model in self.fitted_models:
            est = model['model']
            if is_classification_task(self.task):
                preds = est.predict_proba(X)
            else:
                preds = est.predict(X)
            preds_list.append(np.asarray(preds))
        return preds_list
