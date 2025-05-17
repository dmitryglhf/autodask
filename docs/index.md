# AutoDask

<div align="center">

<img title="AutoDask Logo" alt="AutoDask Logo" src="/img/logo.png" width="150">

</div>

> **Note:** AutoDask is currently in the development stage, which is why some modules may not work or may have errors.

## The Distributed AutoML Solution

AutoDask is a lightweight AutoML library that brings together the power of distributed computing with Dask and the intelligence of Bee Colony Optimization for hyperparameter tuning.

```commandline
pip install autodask
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

## Why Choose AutoDask?

### 🚀 Distributed Performance
Built on Dask to scale from your laptop to your cluster with minimal configuration changes.

### 🐝 Bio-Inspired Optimization
Uses Bee Colony Optimization to intelligently navigate the vast hyperparameter space with fewer iterations than traditional methods.

### 🧠 Smart Resource Management
Automatically allocates computational resources for maximum efficiency during model training and evaluation.

### 🔄 End-to-End Pipeline
Handles everything from preprocessing and feature engineering to model selection and ensemble creation.

## Core Features

- **Multi-Task Support:** Classification and regression workflows
- **Distributed Computing:** Parallel model training and evaluation
- **Automated Feature Engineering:** Intelligent preprocessing and transformation
- **Hyperparameter Optimization:** Nature-inspired BCO algorithm
- **Model Ensembling:** Combines top-performing models for improved accuracy
- **Time Management:** Set time constraints for your AutoML workflows

## Quick Links

- [Basic Tutorial](./autodask/basic_usage.md)
- [API Reference](./autodask/main.md)
- [Bee Colony Optimization Details](./autodask/tuner.md)

## Example Use Cases

Coming soon...
