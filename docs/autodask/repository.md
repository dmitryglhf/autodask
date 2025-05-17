# ML Models and Metrics Repository

## Overview

AutoDask provides standardized access to machine learning models and evaluation metrics with the following key components:

1. **AtomizedModel** class - Repository of machine learning models with default hyperparameters and hyperparameter search spaces
2. **Metrics Repository** - Collection of standardized evaluation metrics for classification and regression tasks

## AtomizedModel Class

The `AtomizedModel` class serves as a repository of machine learning models, providing consistent interfaces for accessing various classification and regression algorithms along with their hyperparameter configurations.

### Key Features

- Access to standardized machine learning models
- Default hyperparameters for quick experimentation
- Hyperparameter search spaces for optimization

### Available Models

#### Classification Models

| Model Name | Implementation | Description |
|------------|----------------|-------------|
| `l2_logreg` | `LogisticRegression` | Logistic regression with L2 regularization |
| `extra_trees` | `ExtraTreesClassifier` | Ensemble of extremely randomized trees |
| `lgbm` | `LGBMClassifier` | LightGBM gradient boosting classifier |
| `xgboost` | `XGBClassifier` | XGBoost gradient boosting classifier |
| `catboost` | `CatBoostClassifier` | CatBoost gradient boosting classifier |

#### Regression Models

| Model Name | Implementation | Description |
|------------|----------------|-------------|
| `l2_linreg` | `LinearRegression` | Linear regression with L2 regularization |
| `extra_trees` | `ExtraTreesRegressor` | Ensemble of extremely randomized trees |
| `lgbm` | `LGBMRegressor` | LightGBM gradient boosting regressor |
| `xgboost` | `XGBRegressor` | XGBoost gradient boosting regressor |
| `catboost` | `CatBoostRegressor` | CatBoost gradient boosting regressor |

### Methods

#### `get_classifier_models()`

Returns a dictionary of classification models with their hyperparameter spaces and default configurations.

**Returns:**
- Dictionary with model names as keys and tuples of (model class, hyperparameter search space, default hyperparameters) as values

**Example:**
```python
models = AtomizedModel.get_classifier_models()
classifier = models['lgbm'][0](**models['lgbm'][2])  # Create LGBM with default params
```

#### `get_regressor_models()`

Returns a dictionary of regression models with their hyperparameter spaces and default configurations.

**Returns:**
- Dictionary with model names as keys and tuples of (model class, hyperparameter search space, default hyperparameters) as values

**Example:**
```python
models = AtomizedModel.get_regressor_models()
regressor = models['xgboost'][0](**models['xgboost'][2])  # Create XGBoost with default params
```

#### `_load_parameter_spaces()` (private)

Loads hyperparameter search spaces from a JSON configuration file.

**Returns:**
- Dictionary of parameter search spaces organized by model name

#### `_load_parameter_default()` (private)

Loads default hyperparameters from a JSON configuration file.

**Returns:**
- Dictionary of default parameters organized by model name

## Metrics Repository

The metrics repository provides a standardized way to access evaluation metrics for machine learning models.

### Available Metrics

#### Classification Metrics

| Metric Name | Function | Default | Maximize |
|-------------|----------|---------|----------|
| `accuracy` | `metrics.accuracy_score` | Yes | Yes |
| `f1` | `metrics.f1_score` | No | Yes |
| `precision` | `metrics.precision_score` | No | Yes |
| `recall` | `metrics.recall_score` | No | Yes |

#### Regression Metrics

| Metric Name | Function | Default | Maximize |
|-------------|----------|---------|----------|
| `mse` | `metrics.mean_squared_error` | Yes | No |
| `r2` | `metrics.r2_score` | No | Yes |
| `mae` | `metrics.mean_absolute_error` | No | No |
| `rmse` | Custom (sqrt of MSE) | No | No |

### Functions

#### `get_metric(metric_name, task=None)`

Retrieves a specific metric function and its properties.

**Parameters:**
- `metric_name` (str): Name of the metric to retrieve
- `task` (str, optional): Task type ('classification' or 'regression') for validation

**Returns:**
- Tuple containing (metric function, metric name, whether to maximize)

**Example:**
```python
metric_func, name, maximize = get_metric('f1', task='classification')
score = metric_func(y_true, y_pred)
```

#### `get_default_metric(task)`

Retrieves the default metric for a specific task type.

**Parameters:**
- `task` (str): Task type ('classification' or 'regression')

**Returns:**
- Tuple containing (metric function, metric name, whether to maximize)

**Example:**
```python
metric_func, name, maximize = get_default_metric('regression')
score = metric_func(y_true, y_pred)
```

## Configuration Files

The library uses two JSON configuration files:

1. **search_parameters.json** - Contains hyperparameter search spaces for each model
2. **default_parameters.json** - Contains default hyperparameters for each model

Both files should be located in the same directory as the main module.
