import time

import numpy as np

# import dask.array as da

from core.tuner import BeeColonyOptimizer
from utils.log import get_logger
from repository.model_repository import AtomizedModel
from dask_ml.model_selection import train_test_split

import warnings

from utils.regular_functions import is_classification_task, setup_metric

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self,
                 task: str,
                 with_tuning=None,
                 time_limit=None,
                 metric:str=None,
                 optimization_rounds=30,
                 max_ensemble_models=None,
                 models=None,
                 bco_params=None):
        self.task = task
        self.with_tuning = with_tuning
        self.time_limit = time_limit
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.model_names = models
        self.bco_params = bco_params or {}
        self.start_time = None

        self.log = get_logger(self.__class__.__name__)

        # Setup metric function
        self.score_func, self.metric_name, self.maximize_metric = (
            setup_metric(metric_name=metric, task=task)
        )

    def launch(self, X, y, validation_data=None):
        self.start_time = time.time()

        # Handle validation data
        if validation_data:
            X_val, y_val = validation_data
        else:
            # Split data if validation set not provided
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=101, shuffle=True
            )
            X, X_val = X_train, X_val
            y, y_val = y_train, y_val

        # Get models based on task or provided model names
        models = self._get_models()

        self.log.info('Starting models training')
        fitted_models = []

        # Initialize bee colony optimizer
        bco = BeeColonyOptimizer(**self.bco_params)

        # For each model, optimize hyperparameters and train
        for name, (model_class, param_space, param_default) in models.items():
            if self._check_time_limit():
                self.log.info(f"Time limit reached.")
                break

            self.log.info(f"Fitting {name} model...")

            # Get remaining time for this model
            remaining_time = None
            if self.time_limit:
                elapsed = time.time() - self.start_time
                remaining_time = max(0, self.time_limit - elapsed)

            # Optimize hyperparameters with bee colony algorithm
            if self.with_tuning:
                self.log.info(f"Optimizing {name} model...")
                best_params, best_score = bco.optimize(
                    model_class=model_class,
                    param_space=param_space,
                    X_train=X,
                    y_train=y,
                    X_val=X_val,
                    y_val=y_val,
                    metric_func=self.score_func,
                    maximize=self.maximize_metric,
                    rounds=self.optimization_rounds,
                    time_limit=remaining_time
                )

                self.log.info(f"Obtained optimized parameters: {best_params}")

                # Train the model with best parameters
                model = model_class(**best_params)
            else:
                self.log.info(f"Obtained parameters: {param_default}")
                model = model_class(**param_default)

            model.fit(X, y)

            # Evaluate and log performance
            if is_classification_task(self.task):
                probs = model.predict_proba(X_val)
                y_pred = np.argmax(probs, axis=1)
            else:
                y_pred = model.predict(X_val)

            validation_score = self.score_func(y_val, y_pred)
            self.log.info(f'The {name} model has been successfully fitted. '
                          f'With {self.metric_name} score on validation set: {validation_score}')

            # Store model with its score
            fitted_models.append({
                'model': model,
                'name': name,
                'score': validation_score if self.maximize_metric else -validation_score
            })

        # Sort models by performance
        fitted_models.sort(key=lambda x: x['score'], reverse=True)

        # Return the best ensemble
        if not fitted_models:
            raise ValueError("No models were successfully trained. Check the logs for errors.")

        return fitted_models[:min(len(fitted_models), self.max_ensemble_models)]

    def _check_time_limit(self):
        """Check if time limit has been reached"""
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            return True
        return False

    def _get_models(self):
        """Get models based on task or provided model names"""
        if is_classification_task(self.task):
            all_models = AtomizedModel.get_classifier_models()
        else:
            all_models = AtomizedModel.get_regressor_models()

        # Filter models if specific ones were requested
        if self.model_names:
            models = {name: model for name, model in all_models.items() if name in self.model_names}
            if not models:
                self.log.info(
                    f"None of the specified models {self.model_names} were found. Using all available models.")
                models = all_models
        else:
            models = all_models

        return models
