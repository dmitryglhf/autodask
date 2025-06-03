import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
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


@pytest.fixture
def fitted_classification_models(classification_data):
    """Creates fitted classification models"""
    X, y = classification_data

    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(X, y)

    model2 = LogisticRegression(random_state=42, max_iter=1000)
    model2.fit(X, y)

    return [
        {'model': model1, 'name': 'RandomForest'},
        {'model': model2, 'name': 'LogisticRegression'}
    ]


@pytest.fixture
def fitted_binary_classification_models(binary_classification_data):
    """Creates fitted binary classification models"""
    X, y = binary_classification_data

    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(X, y)

    model2 = LogisticRegression(random_state=42, max_iter=1000)
    model2.fit(X, y)

    return [
        {'model': model1, 'name': 'RandomForest'},
        {'model': model2, 'name': 'LogisticRegression'}
    ]


@pytest.fixture
def fitted_regression_models(regression_data):
    """Creates fitted regression models"""
    X, y = regression_data

    model1 = RandomForestRegressor(n_estimators=10, random_state=42)
    model1.fit(X, y)

    model2 = LinearRegression()
    model2.fit(X, y)

    return [
        {'model': model1, 'name': 'RandomForest'},
        {'model': model2, 'name': 'LinearRegression'}
    ]
