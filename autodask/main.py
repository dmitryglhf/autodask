from typing import Union

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster

from core.ensembling import EnsembleBlender
from core.trainer import Trainer
from utils.log import get_logger


class AutoDask:
    def __init__(
            self,
            task: str,
            n_jobs=4,
            with_tuning=False,
            time_limit=60*5,
            metric:str=None,
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
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.models = models
        self.bco_params = bco_params or {}

        self.ensemble = None
        self.log = get_logger(self.__class__.__name__)

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, np.ndarray],
            validation_data:tuple[Union[pd.DataFrame, np.ndarray]]=None):
        self._create_dask_server()

        trainer = Trainer(
            task=self.task,
            with_tuning=self.with_tuning,
            time_limit=self.time_limit,
            metric=self.metric,
            optimization_rounds=self.optimization_rounds,
            max_ensemble_models=self.max_ensemble_models,
            models=self.models,
            bco_params=self.bco_params
        )

        best_models = trainer.launch(
            X, y,
            validation_data=validation_data,
        )

        self.ensemble = EnsembleBlender(best_models, task=self.task, metric=self.metric)
        self.ensemble.fit(y)

        self._shutdown_dask_server()
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def best_model(self):
        return self.ensemble

    def save(self, path):
        """Implementation for saving the model (unsupported yet)"""
        import pickle

    def load_model(self, path):
        """Implementation for loading the model (unsupported yet)"""
        import pickle

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
        self.log.info('Dashboard is available at http://localhost:8787/status.')

    def _shutdown_dask_server(self):
        self.log.info('Shutting down Dask Server...')
        if self.dask_client is not None:
            self.dask_client.close()
            del self.dask_client
        if self.dask_cluster is not None:
            self.dask_cluster.close()
            del self.dask_cluster
