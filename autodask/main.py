from distributed import Client, LocalCluster
from core.task.train import Trainer
from utils.log import get_logger


class AutoDask:
    def __init__(self, task: str):
        self.task = task
        self.best_model = None
        self.log = get_logger('AutoDask')

    def fit(self, X, y):
        self._create_dask_server()

        trainer = Trainer(task=self.task)
        model_and_score = trainer.launch(X, y)

        model, name = model_and_score['best_model']['model']
        self.log.info(f"Chosen best model: {name}")
        self.best_model = model

        self._shutdown_dask_server()
        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)

    def best_model(self):
        return self.best_model

    def _create_dask_server(self):
        self.log.info('Creating Dask Server...')
        cluster_params =  dict(
            processes=False,
            n_workers=1,
            threads_per_worker=4,
            memory_limit='auto'
        )

        cluster = LocalCluster(**cluster_params)

        self.dask_client = Client(cluster)
        self.dask_cluster = cluster


    def _shutdown_dask_server(self):
        self.log.info('Shutting down Dask Server...')
        if self.dask_client is not None:
            self.dask_client.close()
            del self.dask_client
        if self.dask_cluster is not None:
            self.dask_cluster.close()
            del self.dask_cluster
