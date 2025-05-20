# BeeColonyOptimizer Module Documentation

## Overview

The `BeeColonyOptimizer` class provides an simplified implementation of the Bee Colony Optimization (BCO) algorithm for hyperparameter tuning in machine learning models. This algorithm is inspired by the foraging behavior of honey bees and is effective for exploring complex hyperparameter spaces to find optimal configurations.

## Class: BeeColonyOptimizer

```python
from autodask.core.optimizer import BeeColonyOptimizer
```

### Description

A hyperparameter optimization solution based on the Bee Colony Optimization algorithm. It efficiently explores the hyperparameter space to find optimal model configurations for both classification and regression tasks.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str | *required* | The machine learning task type. Supported values: 'classification', 'regression'. |
| `employed_bees` | int | 3 | Number of employed bees in the colony. These bees explore the neighborhood of existing solutions. |
| `onlooker_bees` | int | 3 | Number of onlooker bees in the colony. These bees focus on promising solutions. |
| `exploration_rate` | float | 0.3 | Balance between exploration (random search) and exploitation (neighborhood refinement). Higher values favor exploration. |
| `cv_folds` | int | 2 | Number of cross-validation folds for evaluating solutions. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `solutions` | list | Current solutions maintained by the colony (each solution is a tuple of parameters and score). |
| `solution_trials` | dict | Tracks how many times each solution has been tried without improvement. |
| `task` | str | The machine learning task type ('classification' or 'regression'). |
| `cv_folds` | int | Number of cross-validation folds. |
| `log` | Logger | Logger instance for tracking progress. |

### Methods

#### `optimize(model_class, param_space, X_train, y_train, metric_func, maximize=True, rounds=1, time_limit=None)`

Run the bee colony optimization algorithm to find optimal hyperparameters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_class` | Class | *required* | The machine learning model class to optimize. |
| `param_space` | dict | *required* | Parameter search space defining possible values for each hyperparameter. |
| `X_train` | Union[pd.DataFrame, np.ndarray] | *required* | Training features. |
| `y_train` | Union[pd.Series, np.ndarray] | *required* | Training target. |
| `metric_func` | callable | *required* | Scoring function to evaluate model performance. |
| `maximize` | bool | True | Whether to maximize or minimize the metric. Set to True for metrics like accuracy, F1; False for metrics like error, loss. |
| `rounds` | int | 1 | Number of optimization rounds to perform. |
| `time_limit` | int | None | Maximum time in seconds for optimization. If None, no time limit is applied. |

**Returns:**

A tuple containing the best parameters found and their corresponding score.


#### `initialize_colony(model_class, param_space, X_train, y_train, metric_func)`

Initialize bee colony with random solutions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_class` | Class | The machine learning model class to optimize. |
| `param_space` | dict | Parameter search space defining possible values for each hyperparameter. |
| `X_train` | Union[pd.DataFrame, np.ndarray] | Training features. |
| `y_train` | Union[pd.Series, np.ndarray] | Training target. |
| `metric_func` | callable | Scoring function to evaluate model performance. |

**Example:**

```python
optimizer.initialize_colony(
    model_class=LGBMClassifier,
    param_space=param_space,
    X_train=X_train,
    y_train=y_train,
    metric_func=accuracy_score
)
```

#### `update_best_solution(best_params, best_score, maximize)`

Updates the best solution found by the colony.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `best_params` | dict | Current best parameters. |
| `best_score` | float | Current best score. |
| `maximize` | bool | Whether to maximize or minimize the score. |

**Returns:**

A tuple containing the updated best parameters and best score.

#### `employed_bees_phase(model_class, param_space, X_train, y_train, metric_func, maximize)`

Executes the employed bees phase where bees explore the neighborhood of existing solutions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_class` | Class | The machine learning model class to optimize. |
| `param_space` | dict | Parameter search space defining possible values for each hyperparameter. |
| `X_train` | Union[pd.DataFrame, np.ndarray] | Training features. |
| `y_train` | Union[pd.Series, np.ndarray] | Training target. |
| `metric_func` | callable | Scoring function to evaluate model performance. |
| `maximize` | bool | Whether to maximize or minimize the score. |

#### `onlooker_bees_phase(model_class, param_space, X_train, y_train, metric_func, maximize)`

Executes the onlooker bees phase where bees focus on promising solutions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_class` | Class | The machine learning model class to optimize. |
| `param_space` | dict | Parameter search space defining possible values for each hyperparameter. |
| `X_train` | Union[pd.DataFrame, np.ndarray] | Training features. |
| `y_train` | Union[pd.Series, np.ndarray] | Training target. |
| `metric_func` | callable | Scoring function to evaluate model performance. |
| `maximize` | bool | Whether to maximize or minimize the score. |

### Private Methods

These methods are used internally by the BeeColonyOptimizer class:

- `_calculate_selection_probabilities(maximize)`: Calculates probability distribution for solution selection by onlooker bees.
- `_get_params_key(params)`: Creates a stable hashable key for parameter dictionaries.
- `_generate_random_params(param_space)`: Generates random parameters from the parameter space.
- `_explore_neighborhood(params, param_space)`: Generates a new solution by exploring neighborhood of current solution.

## Usage Examples

### Basic Hyperparameter Optimization for Classification

```python
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import ExtraTreesClassifier
from autodask.core.tuner import BeeColonyOptimizer


xt_params_space = {
    "n_estimators": [50, 100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [False, True]
}

bco = BeeColonyOptimizer(task='classification')

best_params, best_score = bco.optimize(
    model_class=ExtraTreesClassifier,
    param_space=xt_params_space,
    X_train=X_train,
    y_train=y_train,
    metric_func=accuracy,
    maximize=True,
    rounds=1,
    time_limit=None
)

print(f'Best score: {best_score}')
print(f'Best parameters: \n {best_params}')

# Now you can use optimized parameters...
```

## Notes

- The `BeeColonyOptimizer` uses cross-validation to evaluate each solution, making it robust to overfitting.
- A large number of rounds can take a very long time to process
