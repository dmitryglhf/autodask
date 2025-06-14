import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def classification_data():
    """Creates test data for classification"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                               n_informative=3, n_redundant=0, random_state=42)
    return X, y


@pytest.fixture
def binary_classification_data():
    """Creates test data for binary classification"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2,
                               n_redundant=0, random_state=42)
    return X, y


@pytest.fixture
def regression_data():
    """Creates test data for regression"""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X, y
