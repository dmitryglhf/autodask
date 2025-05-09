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

adsk = AutoDask(task='classification')
adsk.fit(X_train, y_train)
predictions = adsk.predict(X_test)
```
