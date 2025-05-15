from typing import Union

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster

from joblib import dump, load
from core.blender import WeightedAverageBlender
from core.preprocessor import Preprocessor
from core.trainer import Trainer
from utils.log import get_logger
from utils.regular_functions import is_classification_task, get_n_classes


class AutoDask:
    def __init__(
            self,
            task: str,
            n_jobs=4,
            with_tuning=False,
            time_limit=60*5,
            metric:str=None,
            cv_folds=5,
            seed=101,
            optimization_rounds=30,
            max_ensemble_models=3,
            models=None,
            bco_params=None
    ):
        self.task = task
        self.n_jobs = n_jobs
        self.with_tuning = with_tuning
        self.time_limit = time_limit
        self.metric = metric
        self.cv_folds = cv_folds
        self.seed = seed
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.models = models
        self.bco_params = bco_params or {}

        self.ensemble = None
        self.log = get_logger(self.__class__.__name__)
        self.n_classes = None

        self.prep = None

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.DataFrame, np.ndarray],
            validation_data:tuple[Union[pd.DataFrame, np.ndarray]]=None):
        self._create_dask_server()

        if is_classification_task(self.task):
            self.n_classes = get_n_classes(y_train)
            if self.n_classes == 2:
                self.log.info(f"Task: binary {self.task}")
            elif self.n_classes > 2:
                self.log.info(f"Task: multiclass {self.task}")
            else:
                raise ValueError(f"Obtained {self.n_classes}. Unable to classify.")

        self.log.info("Obtained constraints:")
        self.log.info(f"time: {self.time_limit:.2f} seconds")
        self.log.info(f"CPU: {self.n_jobs} cores")
        if self.models is not None:
            self.log.info(f"Models to be considered: {self.models}")

        self.prep = Preprocessor(encoding='onehot', target_encoding=True)
        X_train_prep, y_train_enc = self.prep.fit_transform(X_train, y_train)
        self.log.info('Features preprocessing finished')

        trainer = Trainer(
            task=self.task,
            with_tuning=self.with_tuning,
            time_limit=self.time_limit,
            metric=self.metric,
            cv_folds=self.cv_folds,
            seed=self.seed,
            optimization_rounds=self.optimization_rounds,
            max_ensemble_models=self.max_ensemble_models,
            models=self.models,
            bco_params=self.bco_params
        )

        best_models = trainer.launch(
            X_train_prep, y_train_enc,
            validation_data=validation_data,
        )

        self.ensemble = WeightedAverageBlender(
            best_models,
            task=self.task,
            metric=self.metric,
            n_classes=self.n_classes
        )
        self.ensemble.fit(X_train_prep, y_train_enc)

        self._shutdown_dask_server()
        return self

    def predict(self, X_test):
        X_test_prep = self.prep.transform(X_test)
        y_pred_enc = self.ensemble.predict(X_test_prep)
        return self.prep.inverse_transform_target(y_pred_enc)

    def predict_proba(self, X_test):
        X_test_prep = self.prep.transform(X_test)
        return self.ensemble.predict_proba(X_test_prep)

    def fitted_ensemble(self):
        return self.ensemble

    def save(self, path):
        """Implementation for saving the ensemble"""
        dump(self.ensemble, path)

    def load_model(self, path):
        """Implementation for loading the ensemble"""
        self.ensemble = load(path)

    def _create_dask_server(self):
        self.log.info('Creating Dask Server...')

        # Determine number of workers based on n_jobs
        n_workers = 1
        if self.n_jobs is not None:
            if self.n_jobs == -1:
                import os
                n_workers = os.cpu_count()
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
        self.log.info('Dashboard is available at http://localhost:8787/status')

    def _shutdown_dask_server(self):
        self.log.info('Shutting down Dask Server...')
        if self.dask_client is not None:
            self.dask_client.close()
            del self.dask_client
        if self.dask_cluster is not None:
            self.dask_cluster.close()
            del self.dask_cluster
