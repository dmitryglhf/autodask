from autodask.core.train import Trainer, Composer


class AutoDask:
    def __init__(self, task: str, cpu: int):
        self.task = task
        self.cpu = cpu
        self.fitted_ensemble = None

    def fit(self, X, y):
        trainer = Trainer(task=self.task)
        fitted_models = trainer.launch(X, y)

        ensembler = Composer()
        self.fitted_ensemble = ensembler.compose_and_predict(
            fitted_models
        )

    def predict(self, X):
        pass