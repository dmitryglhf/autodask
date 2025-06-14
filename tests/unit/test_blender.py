import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from autodask.core.data import ModelContainer
from autodask.core.blender import WeightedAverageBlender


def test_weighted_average_blender_inheritance():
    """Test that blender inherits from BaseEstimator"""
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    blender = WeightedAverageBlender([ModelContainer(model)], task='classification')
    assert isinstance(blender, BaseEstimator)


def test_weighted_average_blender_init():
    """Test blender initialization"""
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    blender = WeightedAverageBlender(
        model_containers=[ModelContainer(model)],
        task='classification',
        max_iter=50
    )
    assert blender.task == 'classification'
    assert blender.max_iter == 50
    assert blender.weights is None


def test_weighted_average_blender_input_validation():
    """Test input validation for model_containers"""
    with pytest.raises(ValueError, match="model_containers must be a non-empty list"):
        WeightedAverageBlender([], task='classification')

    with pytest.raises(ValueError, match="must be ModelContainer instances"):
        WeightedAverageBlender([LogisticRegression()], task='classification')


def test_weighted_average_blender_models_validation(classification_data):
    """Test that all models must be fitted and task-compatible"""
    X, y = classification_data

    # Unfitted model case
    unfitted_model = LogisticRegression()
    with pytest.raises(ValueError, match="is not fitted"):
        WeightedAverageBlender([ModelContainer(unfitted_model)], task='classification').fit(X, y)

    # Wrong task case
    fitted_clf = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="but blender is for task 'regression'"):
        WeightedAverageBlender([ModelContainer(fitted_clf)], task='regression').fit(X, y)


def test_weighted_average_blender_single_model(classification_data):
    """Test blender with single model (should use weight 1.0)"""
    X, y = classification_data
    model = LogisticRegression().fit(X, y)
    blender = WeightedAverageBlender([ModelContainer(model)], task='classification').fit(X, y)

    assert len(blender.weights) == 1
    assert blender.weights[0] == 1.0
    assert np.array_equal(blender.predict(X), model.predict(X))


def test_weighted_average_blender_classification(classification_data):
    """Test blender for classification task"""
    X, y = classification_data
    model1 = LogisticRegression().fit(X, y)
    model2 = RandomForestClassifier().fit(X, y)

    blender = WeightedAverageBlender(
        [ModelContainer(model1), ModelContainer(model2)],
        task='classification'
    ).fit(X, y)

    assert len(blender.weights) == 2
    assert np.isclose(np.sum(blender.weights), 1.0)
    assert all(0 <= w <= 1 for w in blender.weights)

    preds = blender.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1, 2})  # 3 classes in classification_data


def test_weighted_average_blender_binary_classification(binary_classification_data):
    """Test blender for binary classification task"""
    X, y = binary_classification_data
    model1 = LogisticRegression().fit(X, y)
    model2 = RandomForestClassifier().fit(X, y)

    blender = WeightedAverageBlender(
        [ModelContainer(model1), ModelContainer(model2)],
        task='classification'
    ).fit(X, y)

    preds = blender.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})

    proba = blender.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_weighted_average_blender_regression(regression_data):
    """Test blender for regression task"""
    X, y = regression_data
    model1 = LinearRegression().fit(X, y)
    model2 = RandomForestRegressor().fit(X, y)

    blender = WeightedAverageBlender(
        [ModelContainer(model1), ModelContainer(model2)],
        task='regression'
    ).fit(X, y)

    assert len(blender.weights) == 2
    assert np.isclose(np.sum(blender.weights), 1.0)

    preds = blender.predict(X)
    assert preds.shape == y.shape


def test_weighted_average_blender_not_fitted_error(classification_data):
    """Test that blender raises NotFittedError when used before fitting"""
    X, y = classification_data
    model = LogisticRegression().fit(X, y)
    blender = WeightedAverageBlender([ModelContainer(model)], task='classification')

    with pytest.raises(NotFittedError):
        blender.predict(X)

    with pytest.raises(NotFittedError):
        blender.predict_proba(X)


def test_weighted_average_blender_predict_proba_error(regression_data):
    """Test that predict_proba raises error for regression tasks"""
    X, y = regression_data
    model = LinearRegression().fit(X, y)
    blender = WeightedAverageBlender([ModelContainer(model)], task='regression').fit(X, y)

    with pytest.raises(ValueError, match="predict_proba is only available for classification tasks"):
        blender.predict_proba(X)


def test_weighted_average_blender_fit_predict(classification_data):
    """Test fit_predict method"""
    X, y = classification_data
    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.5).fit(X, y)

    blender = WeightedAverageBlender(
        [ModelContainer(model1), ModelContainer(model2)],
        task='classification'
    )
    preds = blender.fit_predict(X, y)
    assert preds.shape == y.shape


def test_weighted_average_blender_fit_predict_proba(binary_classification_data):
    """Test fit_predict_proba method"""
    X, y = binary_classification_data
    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.5).fit(X, y)

    blender = WeightedAverageBlender(
        [ModelContainer(model1), ModelContainer(model2)],
        task='classification'
    )
    proba = blender.fit_predict_proba(X, y)
    assert proba.shape == (len(y), 2)
