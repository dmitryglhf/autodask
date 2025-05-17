# Using Custom BCO Parameters

```python
from autodask.main import AutoDask

# Configure with custom BCO parameters
adsk = AutoDask(
    task='classification',
    with_tuning=True,
    bco_params={
        'employed_bees': 20,     # Number of employed bees
        'onlooker_bees': 10,     # Number of onlooker bees
        'scout_bees': 5,         # Number of scout bees
        'abandonment_limit': 10, # Limit before abandoning a solution
        'exploration_rate': 0.3  # Balance between exploration and exploitation
    }
)

# Train the model
adsk.fit(X_train, y_train)
```
