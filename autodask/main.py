from core.task.train import Trainer
from core.utils.log import get_logger


class AutoDask:
    def __init__(self, task: str):
        self.task = task
        self.best_model = None
        self.log = get_logger('Trainer')

    def fit(self, X, y):
        trainer = Trainer(task=self.task)
        dict_model_score = trainer.launch(X, y)

        model, _ = dict_model_score['best_model'].items()
        self.log.info(f"Chosen best model: {model.__class__.__name__}")
        self.best_model = model

        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)
