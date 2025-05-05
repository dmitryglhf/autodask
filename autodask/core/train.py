from autodask.core.log import get_logger
from autodask.models.model_repository import AtomizedModel


class Trainer:
    def __init__(self, task: str):
        self.task = task
        self.logger = get_logger('Trainer')

    def launch(self, X, y):
        models = {}
        if self.task == 'classification':
            models = AtomizedModel.CLF_MODELS
        elif self.task == 'regression':
            models = AtomizedModel.REG_MODELS
        else:
            raise ValueError('Unexpected task')

        self.logger.log(45, 'Starting models training')
        fitted_models = []
        for name, model in models.value:
            model.fit(X, y)
            self.logger.log(45, f'The {name} model has been successfully fitted')
            fitted_models.append((name, model))

        return fitted_models


class Composer:
    def __init__(self):
        pass

    def compose_and_predict(self, fitted_models: list):
        pass