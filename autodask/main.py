from typing import Union

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster

from pickle import dump, load
from autodask.core.blender import WeightedAverageBlender
from autodask.core.preprocessor import Preprocessor
from autodask.core.trainer import Trainer
from autodask.utils.log import get_logger
from autodask.utils.regular_functions import is_classification_task, get_n_classes


class AutoDask:
    """Orchestrating class for AutoDask.

    Args:
        task (str): The machine learning task type. Supported values: 'classification', 'regression'.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 4.
        with_tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
        time_limit (int, optional): Maximum time in seconds for the automl process. Defaults to 300 (5 minutes).
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 101.
        optimization_rounds (int, optional): Number of optimization rounds for hyperparameter tuning. Defaults to 30.
        max_ensemble_models (int, optional): Maximum number of models in final ensemble. Defaults to 3.
        preprocess (bool, optional): Whether to apply automatic preprocessing. Defaults to True.
        models (list, optional): Custom list of models to consider. If None, uses default model library.
        bco_params (dict, optional): Parameters for bee colony optimization. Defaults to empty dict.

    Attributes:
        ensemble: The final ensemble model after fitting.
        n_classes: Number of classes (for classification tasks).
        preprocessor: Fitted preprocessing pipeline.
        log: Logger instance for tracking progress.

    Example:

        >>> adsk = AutoDask(task='classification', n_jobs=4, with_tuning=True)
        >>> adsk.fit(X_train, y_train)
        >>> predictions = adsk.predict(X_test)

    """

    def __init__(
            self,
            task: str,
            n_jobs=1,
            with_tuning=False,
            time_limit=60*5,
            cv_folds=5,
            seed=42,
            optimization_rounds=1,
            max_ensemble_models=3,
            preprocess=True,
            models=None,
            bco_params=None
    ):
        self.task = task
        self.n_jobs = n_jobs
        self.with_tuning = with_tuning
        self.time_limit = time_limit
        self.cv_folds = cv_folds
        self.seed = seed
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.models = models
        self.bco_params = bco_params or {}
        self.ensemble = None
        self.n_classes = None

        self.preprocess = preprocess
        self.preprocessor = None

        self.log = get_logger(self.__class__.__name__)

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray, tuple, dict, list],
            y_train: Union[pd.DataFrame, np.ndarray, tuple, dict, list, str]):
        """Train the AutoDask model on the given training data.

        Args:
            X_train: Training features. Can be:
                - pandas DataFrame
                - numpy array
                - tuple/dict/list of arrays (for multi-input models)
            y_train: Training target. Can be:
                - pandas DataFrame/Series
                - numpy array
                - column name (str) if X_train is DataFrame

        Returns:
            self: Returns the instance itself.
        """
        self._create_dask_server()

        self.log.info("Obtained constraints:")
        self.log.info(f"time: {self.time_limit:.2f} seconds")
        self.log.info(f"CPU: {self.n_jobs} cores")
        if self.models is not None:
            self.log.info(f"Models to be considered: {self.models}")

        self._kind_clf(y_train)

        if self.preprocess:
            self.preprocessor = Preprocessor()
            X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
            self.log.info('Features preprocessing finished')

        trainer = Trainer(
            task=self.task,
            time_limit=self.time_limit,
            cv_folds=self.cv_folds,
            seed=self.seed,
            max_ensemble_models=self.max_ensemble_models,
            models=self.models,
        )

        best_models = trainer.launch(X_train, y_train)

        self.ensemble = WeightedAverageBlender(
            best_models,
            task=self.task
        )
        self.ensemble.fit(X_train, y_train)

        self._shutdown_dask_server()
        return self

    def predict(self, X_test):
        """Make predictions on new data using the trained ensemble"""
        if self.preprocessor:
            X_test_prep = self.preprocessor.transform(X_test)
            y_pred_enc = self.ensemble.predict(X_test_prep)
            return self.preprocessor.decode_target(y_pred_enc)
        else:
            return self.ensemble.predict(X_test)

    def predict_proba(self, X_test):
        """Make predictions on new data using the trained ensemble"""
        if self.preprocessor:
            X_test_prep = self.preprocessor.transform(X_test)
            return self.ensemble.predict_proba(X_test_prep)
        else:
            self.ensemble.predict_proba(X_test)

    def get_fitted_ensemble(self):
        """Get the trained ensemble model"""
        if self.ensemble:
            return self.ensemble

    def get_fitted_preprocessor(self):
        """Get the fitted preprocessing pipeline"""
        if self.preprocessor:
            return self.preprocessor

    def save(self, path):
        """Save the trained ensemble to disk using joblib.

        Args:
            path: File path to save the model to.

        Example:
            >>> adsk.save('adsk_model.pkl')
        """
        dump(self.ensemble, path)

    def load_model(self, path):
        """Load a previously saved ensemble from disk.

        Args:
            path: File path to load the model from.

        Example:
            >>> adsk.load_model('adsk_model.pkl')
            >>> adsk.predict(X_new)  # Can now make predictions
        """
        self.ensemble = load(path)

    def _create_dask_server(self):
        self.log.info('Creating Dask Server...')

        # Determine number of workers based on n_jobs
        n_workers = 1
        if self.n_jobs is not None:
            if self.n_jobs == -1:
                import os
                n_workers = os.cpu_count()
                self.n_jobs = n_workers
            elif self.n_jobs > 0:
                n_workers = self.n_jobs

        cluster_params = dict(
            processes=False,
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit='auto'
        )

        cluster = LocalCluster(**cluster_params)
        self.dask_client = Client(cluster)
        self.dask_cluster = cluster
        if cluster: self.log.info('Dask Server successfully created')
        self.log.info('Dashboard by default is available at http://localhost:8787/status')

    def _shutdown_dask_server(self):
        self.log.info('Shutting down Dask Server...')
        if self.dask_client is not None:
            self.dask_client.close()
            del self.dask_client
        if self.dask_cluster is not None:
            self.dask_cluster.close()
            del self.dask_cluster

    def _kind_clf(self, y_train):
        if is_classification_task(self.task):
            self.n_classes = get_n_classes(y_train)
            if self.n_classes == 2:
                self.log.info(f"Task: binary {self.task}")
            elif self.n_classes > 2:
                self.log.info(f"Task: multiclass {self.task}")
            else:
                raise ValueError(f"Obtained {self.n_classes}. Unable to classify.")
            return True
        return False
