import time
from typing import Union

import numpy as np
import pandas as pd

# import dask.array as da
from dask import delayed, compute
from sklearn.model_selection import StratifiedKFold, KFold

from autodask.core.data import ModelContainer
from autodask.utils.log import get_logger
from autodask.repository.model_repository import AtomizedModel

import warnings

from autodask.utils.regular_functions import is_classification_task, setup_metric

warnings.filterwarnings('ignore')


def evaluate_model(
        model_container: ModelContainer,
        params, X, y,
        metric_func,
        task: str,
        cv_folds=5):
    log = get_logger('CrossValidation')
    model_class = model_container.model

    try:
        # Initialize K-fold cross-validator
        kf = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
              if is_classification_task(task)
              else KFold(n_splits=cv_folds, shuffle=True, random_state=42))

        scores = []

        # Initialize OOF predictions array
        if is_classification_task(task):
            n_classes = len(np.unique(y))
            oof_preds = np.zeros((len(y), n_classes))
        else:
            oof_preds = np.zeros(len(y))

        # Perform k-fold cross-validation
        for train_index, val_index in kf.split(X, y if is_classification_task(task) else None):
            X_train = X.iloc[train_index] if hasattr(X, 'iloc') else X[train_index]
            X_val = X.iloc[val_index] if hasattr(X, 'iloc') else X[val_index]
            y_train = y.iloc[train_index] if hasattr(y, 'iloc') else y[train_index]
            y_val = np.ravel(y.iloc[val_index] if hasattr(y, 'iloc') else y[val_index])

            # Train and evaluate model on this fold
            model = model_class(**params)
            model.fit(X_train, y_train)

            if is_classification_task(task):
                y_pred = model.predict_proba(X_val)
            else:
                y_pred = model.predict(X_val)
                y_pred = np.ravel(y_pred)

            # Store OOF predictions
            oof_preds[val_index] = y_pred

            # Calculate fold score
            fold_score = metric_func(y_val, y_pred)
            scores.append(fold_score)

        # Calculate mean score
        mean_score = np.mean(scores)
        log.debug(f"Mean {cv_folds}-fold CV score for params {params}: {mean_score}")

        model_container.update_metrics({'cv_score': mean_score})
        model_container.oof_preds = oof_preds
    except Exception as e:
        log.error(f"Error during evaluation: {str(e)}")
        raise ValueError("Invalid parameters for evaluating")


class Trainer:

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
        """Execute the complete training pipeline."""
        self.start_time = time.time()

        # Get models based on task or provided model names
        model_containers = self._get_models()

        self.log.info(f'Starting models training (cv_folds = {self.cv_folds})')

        # For each model, optimize hyperparameters and train
        def train_single_model(model_container):
            model_name = model_container.model_name
            if self._check_time_limit():
                self.log.info(f"Time limit reached for model {model_name}")
                return None

            self.log.info(f"Training {model_name} model...")

            evaluate_model(
                model_container,
                model_container.hyperparameters,
                X_train, y_train,
                self.score_func, self.task,
                self.cv_folds
            )
            cv_score = model_container.metrics.get('cv_score', None)

            self.log.info(f'{model_name} mean {self.metric_name} cv-score = '
                          f'{cv_score:.5f}')

            final_model = model_container.model(**model_container.hyperparameters)
            final_model.fit(X_train, y_train)

            # Update the model container with the trained model
            model_container.model = final_model

            return model_container

        delayed_tasks = [
            delayed(train_single_model)(mc)
            for mc in model_containers
        ]

        results = compute(*delayed_tasks, scheduler='threads')

        # Remove any failed/None models
        fitted_models = [res for res in results if res is not None]

        # Sort models by performance
        fitted_models.sort(key=lambda x: x.metrics.get('cv_score', None), reverse=False)

        if not fitted_models:
            raise ValueError("No models were successfully trained.")

        self.log.info("Model Ranking:")
        self.log.info(f"{'Rank':<5} {'Name':<15} {'Score':<10}")
        self.log.info("-" * 32)

        for idx, mc in enumerate(fitted_models[:len(fitted_models)], start=1):
            name = mc.model_name
            score = mc.metrics.get('cv_score', 0)
            self.log.info(f"{idx:<5} {name:<15} {score:<10.5f}")

        return fitted_models[:self.max_ensemble_models]

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
            models = [mc for mc in all_models if mc.model_name in self.model_names]
            if not models:
                self.log.info(
                    f"None of the specified models {self.model_names} were found. Using all available models.")
                models = all_models
        else:
            models = all_models

        return models
