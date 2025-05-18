# Orchestrating class for AutoDask

## Class: AutoDask

```python
from autodask.main import AutoDask
```

### Description

An end-to-end automated machine learning solution that handles preprocessing, model selection, hyperparameter tuning, and ensemble construction. It supports both classification and regression tasks with parallel execution using Dask.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str | *required* | The machine learning task type. Supported values: 'classification', 'regression'. |
| `n_jobs` | int | 4 | Number of parallel jobs to run. Set to -1 to use all available cores. |
| `with_tuning` | bool | False | Whether to perform hyperparameter tuning using the Bee Colony Optimization algorithm. |
| `time_limit` | int | 300 | Maximum time in seconds for the AutoML process (default: 5 minutes). |
| `metric` | str | None | Evaluation metric to optimize. If None, defaults to a task-appropriate metric (accuracy for classification, MSE for regression). |
| `cv_folds` | int | 5 | Number of cross-validation folds. |
| `seed` | int | 101 | Random seed for reproducibility. |
| `optimization_rounds` | int | 30 | Number of optimization rounds for hyperparameter tuning. |
| `max_ensemble_models` | int | 3 | Maximum number of models in the final ensemble. |
| `preprocess` | bool | True | Whether to apply automatic preprocessing to the input data. |
| `models` | list | None | Custom list of models to consider. If None, uses the default model library. |
| `bco_params` | dict | {} | Parameters for bee colony optimization. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ensemble` | WeightedAverageBlender | The final ensemble model after fitting. |
| `n_classes` | int | Number of classes (for classification tasks). |
| `preprocessor` | Preprocessor | Fitted preprocessing pipeline. |
| `log` | Logger | Logger instance for tracking progress. |

### Methods

#### `fit(X_train, y_train, validation_data=None)`

Train the AutoDask model on the given training data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | Union[pd.DataFrame, np.ndarray, tuple, dict, list] | *required* | Training features. Can be a pandas DataFrame, numpy array, or a tuple/dict/list of arrays (for multi-input models). |
| `y_train` | Union[pd.DataFrame, np.ndarray, tuple, dict, list, str] | *required* | Training target. Can be a pandas DataFrame/Series, numpy array, or column name (str) if X_train is a DataFrame. |

**Returns:**

The instance itself (self), allowing for method chaining.

**Example:**

```python
# Without validation data (uses k-fold cross-validation)
adsk = AutoDask(task='classification')
adsk.fit(X_train, y_train)
```

#### `predict(X_test)`

Make predictions on new data using the trained ensemble.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_test` | Union[pd.DataFrame, np.ndarray, tuple, dict, list] | Input features to predict on. Should have the same format as X_train in fit(). |

**Returns:**

Array of predictions. For classification, returns class labels. For regression, returns continuous values.

**Example:**

```python
predictions = adsk.predict(X_test)
```

#### `predict_proba(X_test)`

For classification tasks, returns the class probabilities for each sample.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_test` | Union[pd.DataFrame, np.ndarray, tuple, dict, list] | Input features to predict on. Should have the same format as X_train in fit(). |

**Returns:**

Array of prediction probabilities for each class.

**Example:**

```python
probabilities = adsk.predict_proba(X_test)
```

#### `get_fitted_ensemble()`

Get the trained ensemble model.

**Returns:**

The fitted `WeightedAverageBlender` ensemble model.

**Example:**

```python
ensemble = adsk.get_fitted_ensemble()
```

#### `get_fitted_preprocessor()`

Get the fitted preprocessing pipeline.

**Returns:**

The fitted `Preprocessor` preprocessing pipeline.

**Example:**

```python
preprocessor = adsk.get_fitted_preprocessor()
```

#### `save(path)`

Save the trained ensemble to disk.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File path to save the model to. |

**Example:**

```python
adsk.save('adsk_model.pkl')
```

#### `load_model(path)`

Load a previously saved ensemble from disk.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File path to load the model from. |

**Example:**

```python
adsk.load_model('adsk_model.pkl')
adsk.predict(X_new)  # Can now make predictions
```

### Private Methods

These methods are used internally by the AutoDask class:

- `_create_dask_server()`: Initializes the Dask distributed computing environment.
- `_shutdown_dask_server()`: Closes the Dask client and cluster.
- `_check_input_correctness(X, y)`: Validates and formats the input data.
- `_kind_clf(y_train)`: Determines the classification type (binary or multiclass) and sets `n_classes`.


## Notes

- The `AutoDask` class automatically handles distributed computing setup using Dask, with a local dashboard available at http://localhost:8787/status during training.
- When `preprocess=True`, input data is automatically preprocessed, handling categorical features, missing values, and scaling.
- For classification tasks, the class automatically determines if it's binary or multiclass.
- The `time_limit` parameter ensures the AutoML process completes within the specified time constraint.
- The Bee Colony Optimization algorithm is used for hyperparameter tuning when `with_tuning=True`.
