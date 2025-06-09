import time
from typing import Union

import numpy as np
import pandas as pd

# import dask.array as da
from dask import delayed, compute

from autodask.utils.log import get_logger
from autodask.utils.cross_val import evaluate_model
from autodask.repository.model_repository import AtomizedModel

import warnings

from autodask.utils.regular_functions import is_classification_task, setup_metric

warnings.filterwarnings('ignore')


class Trainer:
    """Orchestrates model training, hyperparameter tuning and evaluation.

    This class handles the complete model training pipeline including:
    - Model selection based on task type
    - Hyperparameter optimization
    - Cross-validation
    - Model evaluation and ranking

    Args:
        task (str): Machine learning task type ('classification' or 'regression')
        with_tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to None.
        time_limit (int, optional): Maximum training time in seconds. Defaults to None.
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
                 time_limit=None,
                 cv_folds:int=5,
                 seed:int=None,
                 max_ensemble_models=None,
                 models=None):
        self.task = task
        self.time_limit = time_limit
        self.cv_folds = cv_folds
        self.seed = seed
        self.max_ensemble_models = max_ensemble_models
        self.model_names = models

        self.start_time = None

        self.log = get_logger(self.__class__.__name__)

        # Setup metric function
        self.score_func, self.metric_name, self.maximize_metric = setup_metric(task=task)

    def launch(self, X_train: Union[pd.DataFrame], y_train: Union[np.ndarray]):
        """Execute the complete training pipeline.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training targets

        Returns:
            list: Sorted list of dictionaries containing trained models, their names and scores
        """
        self.start_time = time.time()

        # Get models based on task or provided model names
        models = self._get_models()

        self.log.info(f'Starting models training (cv_folds = {self.cv_folds})')

        # For each model, optimize hyperparameters and train
        def train_single_model(name, model_class, param_default):
            if self._check_time_limit():
                self.log.info(f"Time limit reached for model {name}")
                return None

            self.log.info(f"Training {name} model...")

            validation_score = evaluate_model(
                model_class, param_default, X_train, y_train, self.score_func, self.task, self.cv_folds
            )
            self.log.info(f'{name} mean {self.metric_name} cv-score = {validation_score:.5f}')

            final_model = model_class(**param_default)
            final_model.fit(X_train, y_train)

            return {
                'model': final_model,
                'name': name,
                'score': validation_score
            }

        delayed_tasks = [
            delayed(train_single_model)(name, model_class, param_default)
            for name, (model_class, _, param_default) in models.items()
        ]

        results = compute(*delayed_tasks, scheduler='threads')

        # Remove any failed/None models
        fitted_models = [res for res in results if res is not None]

        # Sort models by performance
        fitted_models.sort(key=lambda x: x['score'], reverse=False)

        if not fitted_models:
            raise ValueError("No models were successfully trained.")

        self.log.info("\nModel Ranking:")
        self.log.info(f"{'Rank':<5} {'Name':<15} {'Score':<10}")
        self.log.info("-" * 32)

        for idx, model_info in enumerate(fitted_models[:min(len(fitted_models), self.max_ensemble_models)], start=1):
            name = model_info['name']
            score = model_info['score']
            self.log.info(f"{idx:<5} {name:<15} {score:<10.5f}")

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
