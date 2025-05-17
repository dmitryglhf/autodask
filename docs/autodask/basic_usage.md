# Basic Classification

```python
from autodask.main import AutoDask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train AutoDask instance
adsk = AutoDask(task='classification')
adsk.fit(X_train, y_train)

# Make predictions
predictions = adsk.predict(X_test)
```

# Saving a Model

```python
from autodask.main import AutoDask

# Create a new instance
adsk = AutoDask(task='classification')

# Load a saved model
adsk.save('adsk_model.pkl')
```

# Loading a Saved Model

```python
from autodask.main import AutoDask

# Create a new instance
adsk = AutoDask(task='classification')

# Load a saved model
adsk.load_model('adsk_model.pkl')

# Use the loaded model to make predictions
predictions = adsk.predict(X_new)
```