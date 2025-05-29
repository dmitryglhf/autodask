# WeightedAverageBlender Module

## Overview

The `WeightedAverageBlender` class provides an ensemble method that creates weighted combinations of multiple trained models. This blender optimizes the weights for each model to maximize performance on the given task.

## Class: WeightedAverageBlender

```python
from autodask.core.blender import WeightedAverageBlender
```

### Description

A weighted average ensemble blender that combines predictions from multiple pre-trained models by optimizing their weights to maximize performance. The class supports both classification and regression tasks.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fitted_models` | list | *required* | List of dictionaries containing trained models. Each dictionary should have 'model' (fitted estimator) and 'name' (string identifier) keys. |
| `task` | str | *required* | The machine learning task type. Supported values: 'classification', 'regression'. |
| `max_iter` | int | 100 | Maximum number of iterations for weight optimization. |
| `tol` | float | 1e-6 | Tolerance for convergence in weight optimization. |
| `n_classes` | int | None | Number of classes for classification tasks. Required for classification, ignored for regression. |

### Attributes

| Attribute | Type | Description                                                         |
|-----------|------|---------------------------------------------------------------------|
| `weights` | np.ndarray | Optimized weights for each model after fitting. Weights sum to 1.0. |
| `fitted_models` | list | List of dictionaries containing the trained models and their names. |
| `task` | str | The machine learning task type ('classification' or 'regression').  |
| `log` | Logger | Logger instance for tracking progress and optimization details.     |
| `score_func` | callable | Default scoring function used for weight optimization.              |
| `metric_name` | str | Name of the metric being optimized.                                 |
| `is_maximize_metric` | bool | Whether the metric should be maximized (True) or minimized (False). |

### Methods

#### `fit(X, y)`

Optimize the weights for ensemble models based on training data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | Union[pd.DataFrame, np.ndarray] | Training features used for weight optimization. |
| `y` | Union[pd.Series, np.ndarray] | Training targets used for weight optimization. |

**Returns:**

The instance itself (self), allowing for method chaining.

**Example:**

```python
# Assuming you have pre-fitted models
models = [
    {'model': fitted_lgbm, 'name': 'LGBM'},
    {'model': fitted_xgb, 'name': 'XGBoost'},
    {'model': fitted_rf, 'name': 'RandomForest'}
]

# Create and fit blender
blender = WeightedAverageBlender(
    fitted_models=models,
    task='classification',
    n_classes=3,
    max_iter=200
)
blender.fit(X_train, y_train)
```

#### `predict(X)`

Make predictions using the weighted ensemble.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | Union[pd.DataFrame, np.ndarray] | Input features to predict on. |

**Returns:**

Array of predictions. For classification, returns class labels. For regression, returns continuous values.

**Example:**

```python
predictions = blender.predict(X_test)
```

#### `predict_proba(X)`

For classification tasks, returns the class probabilities for each sample.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | Union[pd.DataFrame, np.ndarray] | Input features to predict on. |

**Returns:**

Array of prediction probabilities for each class. Only available for classification tasks.

**Example:**

```python
# Only works for classification tasks
probabilities = blender.predict_proba(X_test)
```

#### `fit_predict(X, y)`

Fit the blender and immediately make predictions on the same data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | Union[pd.DataFrame, np.ndarray] | Training features. |
| `y` | Union[pd.Series, np.ndarray] | Training targets. |

**Returns:**

Array of predictions on the training data.

**Example:**

```python
train_predictions = blender.fit_predict(X_train, y_train)
```

#### `fit_predict_proba(X, y)`

Fit the blender and immediately return class probabilities on the same data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | Union[pd.DataFrame, np.ndarray] | Training features. |
| `y` | Union[pd.Series, np.ndarray] | Training targets. |

**Returns:**

Array of prediction probabilities for the training data. Only available for classification tasks.

**Example:**

```python
train_probabilities = blender.fit_predict_proba(X_train, y_train)
```

### Private Methods

These methods are used internally by the WeightedAverageBlender class:

- `_fit(X, y)`: Core optimization logic for finding optimal weights.
- `_blend_predictions(predictions, weights, return_labels)`: Combines predictions using weighted average.
- `_get_model_predictions(X)`: Extracts predictions from all ensemble models.
- `_validate_before_blending()`: Validates that all models are fitted and parameters are correct.

## Usage Examples

### Basic Classification Ensemble

```python
from autodask.core.blender import WeightedAverageBlender
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(random_state=42)
xgb_model = XGBClassifier(random_state=42)

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Create model list for blender
models = [
    {'model': rf_model, 'name': 'RandomForest'},
    {'model': lr_model, 'name': 'LogisticRegression'},
    {'model': xgb_model, 'name': 'XGBoost'}
]

# Create and fit blender
blender = WeightedAverageBlender(
    fitted_models=models,
    task='classification',
    n_classes=3,
    max_iter=150
)

# Fit blender to optimize weights
blender.fit(X_train, y_train)

# Make predictions
predictions = blender.predict(X_test)
probabilities = blender.predict_proba(X_test)

print(f"Optimized weights: {blender.weights}")
```

## Model Dictionary Format

The `fitted_models` parameter expects a list of dictionaries with the following structure:

```python
models = [
    {
        'model': fitted_sklearn_compatible_model,  # Must be already fitted
        'name': 'descriptive_model_name'         # String identifier
    },
    # ... more models
]
```

## Notes

- All models in the ensemble must be pre-fitted before passing to the blender.
- Single-model ensembles are handled gracefully with a weight of 1.0.
- Each model must be already fitted (trained)
