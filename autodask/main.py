from distributed import Client, LocalCluster
from core.trainer import Trainer
from utils.log import get_logger


class AutoDask:
    def __init__(
            self,
            task: str,
            n_jobs=4,
            time_limit=60*5,
            metric=None,
            optimization_rounds=30,
            max_ensemble_models=3,
            models=None,
            bco_params=None,
            verbose=1
    ):
        self.task = task
        self.n_jobs = n_jobs
        self.time_limit = time_limit
        self.metric = metric
        self.optimization_rounds = optimization_rounds
        self.max_ensemble_models = max_ensemble_models
        self.models = models
        self.bco_params = bco_params or {}
        self.verbose = verbose

        self.ensemble = None
        self.log = get_logger(self.__class__.__name__)

    def fit(self, X, y, validation_data:tuple=None):
        self._create_dask_server()

        trainer = Trainer(
            task=self.task,
            time_limit=self.time_limit,
            metric=self.metric,
            optimization_rounds=self.optimization_rounds,
            max_ensemble_models=self.max_ensemble_models,
            models=self.models,
            bco_params=self.bco_params,
            verbose=self.verbose
        )

        model_and_score = trainer.launch(
            X, y,
            validation_data=validation_data,
        )

        model, name = model_and_score['best_model']['model']
        self.log.info(f"Chosen best model: {name}")
        self.ensemble = model

        self._shutdown_dask_server()
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def best_model(self):
        return self.best_model

    def save(self, path):
        """Implementation for saving the model"""
        import pickle

    def load_model(self, path):
        """Implementation for loading the model"""
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

    def _shutdown_dask_server(self):
        self.log.info('Shutting down Dask Server...')
        if self.dask_client is not None:
            self.dask_client.close()
            del self.dask_client
        if self.dask_cluster is not None:
            self.dask_cluster.close()
            del self.dask_cluster
