from typing import Any, Dict, Optional


class ModelContainer:
    """A container class for machine learning models with associated metadata and methods."""
    def __init__(
            self,
            model: Any,
            model_name: str = None,
            model_task_type: str = None,
            metrics: Optional[Dict[str, float]] = None,
            hyperparameters: Optional[Dict[str, Any]] = None,
            search_space: Optional[Dict[str, Any]] = None,
            tag: Optional[str] = None
    ):
        """Initialize the ModelContainer with a model and its metadata.

        Args:
            model: The machine learning model.
            model_name: Name of the model.
            model_task_type: Type of task the model is designed for.
            metrics: Dictionary of performance metrics.
            hyperparameters: Dictionary of hyperparameters used by the model.
            search_space: Search space of hyperparameters used by the model.
            tag: Optional tag for additional identification.
        """
        self.model = model
        self.model_name = model_name or model.__class__.__name__
        self.model_task_type = model_task_type
        self.metrics = metrics or {}
        self.hyperparameters = hyperparameters or {}
        self.search_space = search_space
        self.tag = tag or ""

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba unsupported for this model")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_task_type": self.model_task_type,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "search_space": self.search_space,
            "tag": self.tag
        }

    def update_metrics(self, new_metrics: Dict[str, float]):
        self.metrics.update(new_metrics)

    def __str__(self):
        return f"ModelContainer(model_name={self.model_name}"
