import dask.array as da
import time

from core.bco_optimization import BeeColonyOptimizer
from utils.log import get_logger
from autodask.models.model_repository import AtomizedModel
from dask_ml.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self,
                 task: str,
                 time_limit=None,
                 metric=None,
                 optimization_rounds=30,
                 max_ensemble_models=None,
                 models=None,
                 bco_params=None,
                 verbose=1):
        self.task = task
        self.time_limit = time_limit
        self.metric_name = metric
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.model_names = models
        self.bco_params = bco_params or {}
        self.verbose = verbose
        self.start_time = None

        self.log = get_logger(self.__class__.__name__)

        # Setup metric function
        self._setup_metric()

    def launch(self, X, y, validation_data=None):
        self.start_time = time.time()

        # Convert to Dask arrays if not already
        X = da.from_array(X, chunks='auto') if not isinstance(X, da.Array) else X
        y = da.from_array(y, chunks='auto') if not isinstance(y, da.Array) else y

        # Handle validation data
        if validation_data:
            X_val, y_val = validation_data
            X_val = da.from_array(X_val, chunks='auto') if not isinstance(X_val, da.Array) else X_val
            y_val = da.from_array(y_val, chunks='auto') if not isinstance(y_val, da.Array) else y_val
        else:
            # Split data if validation set not provided
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=101, shuffle=True
            )
            X, X_val = X_train, X_val
            y, y_val = y_train, y_val

        # Get models based on task or provided model names
        models = self._get_models()

        self.log.info('Starting models training and optimization')
        fitted_models = []

        # Initialize bee colony optimizer
        bco = BeeColonyOptimizer(**self.bco_params)

        # For each model, optimize hyperparameters and train
        for name, (model_class, param_space) in models.items():
            if self._check_time_limit():
                self.log.info(f"Time limit reached. Stopping optimization.")
                break

            self.log.info(f"Optimizing {name} model...")

            # Get remaining time for this model
            remaining_time = None
            if self.time_limit:
                elapsed = time.time() - self.start_time
                remaining_time = max(0, self.time_limit - elapsed)

            # Optimize hyperparameters with bee colony algorithm
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

            # Handle sample weights during fitting if provided
            model.fit(X, y)

            # Evaluate and log performance
            y_pred = model.predict(X_val)
            validation_score = self.score_func(y_val, y_pred)

            self.log.info(f'The {name} model has been successfully optimized and fitted. '
                          f'With {self.score_name} score on validation set: {validation_score}')

            # Store model with its score
            fitted_models.append({
                'model': (model, name),
                'score': validation_score if self.maximize_metric else -validation_score,
                'params': best_params
            })

        # Sort models by performance
        fitted_models.sort(key=lambda x: x['score'], reverse=True)

        # Create ensemble if requested
        if self.max_ensemble_models and len(fitted_models) > 1:
            ensemble_models = fitted_models[:min(len(fitted_models), self.max_ensemble_models)]
            # Implement ensemble creation logic here
            # ...

        # Return the best model or ensemble
        if not fitted_models:
            raise ValueError("No models were successfully trained. Check the logs for errors.")

        return {'best_model': fitted_models[0]}

    def _check_time_limit(self):
        """Check if time limit has been reached"""
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            return True
        return False

    def _setup_metric(self):
        """Set up the metric function based on task or provided metric name"""
        if self.task == 'classification':
            if self.metric_name == 'accuracy' or self.metric_name is None:
                self.score_func = accuracy_score
                self.score_name = 'accuracy'
            elif self.metric_name == 'f1':
                self.score_func = f1_score
                self.score_name = 'f1'
            else:
                # Default to accuracy for classification
                self.score_func = accuracy_score
                self.score_name = 'accuracy'
            self.maximize_metric = True

        elif self.task == 'regression':
            if self.metric_name == 'mse' or self.metric_name is None:
                self.score_func = mean_squared_error
                self.score_name = 'mse'
                self.maximize_metric = False
            elif self.metric_name == 'r2':
                self.score_func = r2_score
                self.score_name = 'r2'
                self.maximize_metric = True
            elif self.metric_name == 'mae':
                self.score_func = mean_absolute_error
                self.score_name = 'mae'
                self.maximize_metric = False
            else:
                # Default to MSE for regression
                self.score_func = mean_squared_error
                self.score_name = 'mse'
                self.maximize_metric = False
        else:
            raise ValueError(f'Unsupported task: {self.task}')

    def _get_models(self):
        """Get models based on task or provided model names"""
        if self.task == 'classification':
            all_models = AtomizedModel.get_classifier_models()
        elif self.task == 'regression':
            all_models = AtomizedModel.get_regressor_models()
        else:
            raise ValueError(f'Unexpected task: {self.task}')

        # Filter models if specific ones were requested
        if self.model_names:
            models = {name: model for name, model in all_models.items() if name in self.model_names}
            if not models:
                self.log.warning(
                    f"None of the specified models {self.model_names} were found. Using all available models.")
                models = all_models
        else:
            models = all_models

        return models
