# AutoDask

> **Note:** AutoDask is currently in the development stage, which is why some modules may not work or may have errors.

## The Distributed AutoML

AutoDask is a lightweight AutoML library that brings together the power of distributed computing with Dask(https://www.dask.org/) and the intelligence of Bee Colony Optimization for hyperparameter tuning.

```commandline
pip install autodask  <unsupported yet>
```

```python
from autodask.main import AutoDask

# Create an AutoDask instance
adsk = AutoDask(task='classification')

# Train the model
adsk.fit(X_train, y_train)

# Make predictions
predictions = adsk.predict(X_test)
```

## Core Features

- **Multi-Task Support:** Classification and regression workflows
- **Distributed Computing:** Parallel model training and evaluation
- **Automated Feature Engineering:** Intelligent preprocessing and transformation
- **Hyperparameter Optimization:** Nature-inspired Bee Colony Optimization algorithm
- **Model Ensembling:** Combines top-performing models by using weighted average blending

## Quick Links

- [Basic Usage](./autodask/basic_usage.md)
- [API Reference](./autodask/main.md)
- [Bee Colony Optimization Details](./autodask/tuner.md)

## Example Use Cases

Coming soon...
