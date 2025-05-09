import dask.array as da

from core.utils import get_logger
from autodask.models.model_repository import AtomizedModel
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import accuracy_score as accuracy, mean_squared_error as mse

import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, task: str):
        self.task = task
        self.log = get_logger('Trainer')

    def launch(self, X, y):
        X = da.from_array(X, chunks='auto')
        y = da.from_array(y, chunks='auto')

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=101, shuffle=True
        )

        models = self._get_models_from_task()

        self.log.info('Starting models training')
        fitted_model = {'best_model': {
            'model': None,
            'score': 0,
        }}

        for name, model in models.value.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            current_score = self.score(y_val, y_pred)

            self.log.info(f'The {name} model has been successfully fitted. '
                          f'With {self.score_name} score on validation set: {current_score}')

            # Minimizing operation
            current_score = -current_score

            if current_score <= fitted_model['best_model']['score']:
                fitted_model['best_model'] = {
                    'model': (model, name),
                    'score': current_score,
                }

        if fitted_model['best_model']['model'][0] is None:
            raise ValueError("Model not trained. Call `.fit()` first or check input data.")

        return fitted_model

    def _get_models_from_task(self):
        if self.task == 'classification':
            models = AtomizedModel.CLF_MODELS
            self.score = accuracy
            self.score_name = f'{accuracy=}'.split('=')[0]
        elif self.task == 'regression':
            models = AtomizedModel.REG_MODELS
            self.score = mse
            self.score_name = f'{mse=}'.split('=')[0]
        else:
            raise ValueError('Unexpected task')
        return models
