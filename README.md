<div align="center">

<img src="./img/logo.png" alt="logo" width="150"/>

## AutoDask
AutoML Library Based on Dask
</div>

## :zap: Quickstart

```commandline
pip install autodask
```

```python
from autodask.main import AutoDask

ad = AutoDask(task='classification')
ad.fit(X_train, y_train)
predictions = ad.predict(X_test)
```