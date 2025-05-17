import time
from typing import Union

import numpy as np
import pandas as pd

# import dask.array as da

from autodask.core.tuner import BeeColonyOptimizer
from autodask.utils.log import get_logger
from autodask.utils.cross_val import evaluate_model
from autodask.repository.model_repository import AtomizedModel
from sklearn.model_selection import KFold, StratifiedKFold

import warnings

from autodask.utils.regular_functions import is_classification_task, setup_metric

warnings.filterwarnings('ignore')


class Trainer:
    """Orchestrates model training, hyperparameter tuning and evaluation.

    This class handles the complete model training pipeline including:
    - Model selection based on task type
    - Hyperparameter optimization using Bee Colony Algorithm
    - Cross-validation or hold-out validation
    - Model evaluation and ranking

    Args:
        task (str): Machine learning task type ('classification' or 'regression')
        with_tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to None.
        time_limit (int, optional): Maximum training time in seconds. Defaults to None.
        metric (str, optional): Evaluation metric to optimize. Defaults to None.
        cv_folds (int, optional): Number of cross-validation folds. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        optimization_rounds (int, optional): Number of optimization rounds for tuning. Defaults to 30.
        max_ensemble_models (int, optional): Maximum number of models to keep. Defaults to None.
        models (list, optional): Specific models to use. If None, uses all available. Defaults to None.
        bco_params (dict, optional): Parameters for Bee Colony Optimizer. Defaults to None.

    Attributes:
        score_func (callable): Metric scoring function
        metric_name (str): Name of the evaluation metric
        maximize_metric (bool): Whether higher metric values are better
        log (Logger): Logger instance for tracking progress

    Example:
        ```
        >>> trainer = Trainer(task='classification', cv_folds=5)
        >>> # Without validation data - k-fold cv
        >>> models = trainer.launch(X_train, y_train)
        >>> # With validation data - hold-out cv
        >>> models = trainer.launch(X_train, y_train, validation_data=(X_val, y_val))
        ```
    """

    def __init__(self,
                 task: str,
                 with_tuning=None,
                 time_limit=None,
                 metric:str=None,
                 cv_folds:int=None,
                 seed:int=None,
                 optimization_rounds=30,
                 max_ensemble_models=None,
                 models=None,
                 bco_params=None):
        self.task = task
        self.with_tuning = with_tuning
        self.time_limit = time_limit
        self.cv_folds = cv_folds
        self.seed = seed
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

    def launch(self, X_train: Union[pd.DataFrame], y_train: Union[np.ndarray]):
        """Execute the complete training pipeline.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training targets
            validation_data (tuple, optional): Optional (X_val, y_val) for hold-out validation

        Returns:
            list: Sorted list of dictionaries containing trained models, their names and scores
        """
        self.start_time = time.time()
        if is_classification_task(self.task):
            kf = StratifiedKFold(n_splits=self.cv_folds,
                                 shuffle=True,
                                 random_state=self.seed)
        else:
            kf = KFold(n_splits=self.cv_folds,
                       shuffle=True,
                       random_state=self.seed)

        # Get models based on task or provided model names
        models = self._get_models()

        self.log.info(f'Starting models training (cv_folds = {self.cv_folds})')
        fitted_models = []

        # Initialize bee colony optimizer
        bco = BeeColonyOptimizer(**self.bco_params)

        # For each model, optimize hyperparameters and train
        for name, (model_class, param_space, param_default) in models.items():
            if self._check_time_limit():
                self.log.info(f"Time limit reached")
                break

            self.log.info(f"Fitting {name} model...")

            # Get remaining time for current model
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
                    X_train=X_train, y_train=y_train,
                    metric_func=self.score_func,
                    maximize=self.maximize_metric,
                    rounds=self.optimization_rounds,
                    time_limit=remaining_time
                )

                # Train the model with best parameters
                model_params = best_params
            else:
                model_params = param_default

            # Get score by using k-fold cross validation
            validation_score = evaluate_model(
                model_class, model_params, X_train, y_train, self.score_func, self.cv_folds
            )
            self.log.info(f'{name} mean cv-score = {validation_score:.5f}')

            final_model = model_class(**model_params)
            final_model.fit(X_train, y_train)

            # Store model with its score
            fitted_models.append({
                'model': final_model,
                'name': name,
                'score': validation_score if self.maximize_metric else -validation_score
            })

        # Sort models by performance
        fitted_models.sort(key=lambda x: x['score'], reverse=True)

        # Return the best ensemble
        if not fitted_models:
            raise ValueError("No models were successfully trained.")

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
