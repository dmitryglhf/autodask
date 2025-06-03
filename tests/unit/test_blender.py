import pytest
import numpy as np
from unittest.mock import Mock
from sklearn.ensemble import RandomForestClassifier

from autodask.core.blender import WeightedAverageBlender


def test_init_classification(fitted_classification_models):
    """Test initialization for classification"""
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    assert blender.fitted_models == fitted_classification_models
    assert blender.task == 'classification'
    assert blender.n_classes == 3
    assert blender.max_iter == 100
    assert blender.weights is None


def test_init_regression(fitted_regression_models):
    """Test initialization for regression"""
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    assert blender.fitted_models == fitted_regression_models
    assert blender.task == 'regression'
    assert blender.n_classes is None


def test_fit_classification(fitted_classification_models, classification_data):
    """Test fitting for classification"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    result = blender.fit(X, y)

    assert result is blender
    assert blender.weights is not None
    assert len(blender.weights) == len(fitted_classification_models)
    assert np.allclose(np.sum(blender.weights), 1.0)
    assert np.all(blender.weights >= 0)


def test_fit_regression(fitted_regression_models, regression_data):
    """Test fitting for regression"""
    X, y = regression_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    result = blender.fit(X, y)

    assert result is blender
    assert blender.weights is not None
    assert len(blender.weights) == len(fitted_regression_models)
    assert np.allclose(np.sum(blender.weights), 1.0)
    assert np.all(blender.weights >= 0)


def test_fit_single_model(classification_data):
    """Test fitting with single model"""
    X, y = classification_data

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    blender = WeightedAverageBlender(
        fitted_models=[{'model': model, 'name': 'SingleModel'}],
        task='classification',
        n_classes=3
    )

    blender.fit(X, y)

    assert np.array_equal(blender.weights, np.array([1.0]))


def test_predict_classification(fitted_classification_models, classification_data):
    """Test prediction for classification"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    blender.fit(X, y)
    predictions = blender.predict(X)

    assert len(predictions) == len(y)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)


def test_predict_binary_classification(fitted_binary_classification_models, binary_classification_data):
    """Test prediction for binary classification"""
    X, y = binary_classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_binary_classification_models,
        task='classification',
        n_classes=2
    )

    blender.fit(X, y)
    predictions = blender.predict(X)

    assert len(predictions) == len(y)
    assert np.all(np.isin(predictions, [0, 1]))


def test_predict_regression(fitted_regression_models, regression_data):
    """Test prediction for regression"""
    X, y = regression_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    blender.fit(X, y)
    predictions = blender.predict(X)

    assert len(predictions) == len(y)
    assert predictions.dtype in [np.float64, np.float32]


def test_predict_proba_classification(fitted_classification_models, classification_data):
    """Test probability prediction for classification"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    blender.fit(X, y)
    probabilities = blender.predict_proba(X)

    assert probabilities.shape == (len(y), 3)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert np.all(probabilities >= 0)


def test_predict_proba_regression_error(fitted_regression_models, regression_data):
    """Test error when calling predict_proba for regression"""
    X, y = regression_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    blender.fit(X, y)

    with pytest.raises(ValueError, match="predict_proba is only available for classification tasks"):
        blender.predict_proba(X)


def test_fit_predict(fitted_classification_models, classification_data):
    """Test fit_predict method"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    predictions = blender.fit_predict(X, y)

    assert len(predictions) == len(y)
    assert blender.weights is not None


def test_fit_predict_proba(fitted_classification_models, classification_data):
    """Test fit_predict_proba method"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    probabilities = blender.fit_predict_proba(X, y)

    assert probabilities.shape == (len(y), 3)
    assert blender.weights is not None


def test_validate_unfitted_model(classification_data):
    """Test validation with unfitted model"""
    X, y = classification_data

    unfitted_model = RandomForestClassifier()

    blender = WeightedAverageBlender(
        fitted_models=[{'model': unfitted_model, 'name': 'UnfittedModel'}],
        task='classification',
        n_classes=3
    )

    with pytest.raises(ValueError, match="Model 'UnfittedModel' is not fitted"):
        blender.fit(X, y)


def test_validate_missing_n_classes(fitted_classification_models, classification_data):
    """Test validation of missing n_classes for classification"""
    X, y = classification_data

    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification'
    )

    with pytest.raises(ValueError, match="Number of classes \\(n_classes\\) must be specified"):
        blender.fit(X, y)


def test_blend_predictions_classification(fitted_classification_models):
    """Test _blend_predictions method for classification"""
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    pred1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    pred2 = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    predictions = [pred1, pred2]
    weights = np.array([0.6, 0.4])

    result = blender._blend_predictions(predictions, weights, return_labels=True)

    assert len(result) == 2
    assert np.all(result >= 0)
    assert np.all(result < 3)


def test_blend_predictions_regression(fitted_regression_models):
    """Test _blend_predictions method for regression"""
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    pred1 = np.array([1.0, 2.0, 3.0])
    pred2 = np.array([1.5, 2.5, 3.5])
    predictions = [pred1, pred2]
    weights = np.array([0.7, 0.3])

    result = blender._blend_predictions(predictions, weights, return_labels=True)

    expected = 0.7 * pred1 + 0.3 * pred2
    assert np.allclose(result, expected)


def test_get_model_predictions_classification(fitted_classification_models, classification_data):
    """Test _get_model_predictions method for classification"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    predictions = blender._get_model_predictions(X)

    assert len(predictions) == len(fitted_classification_models)
    for pred in predictions:
        assert pred.shape == (len(X), 3)
        assert np.allclose(np.sum(pred, axis=1), 1.0)


def test_get_model_predictions_regression(fitted_regression_models, regression_data):
    """Test _get_model_predictions method for regression"""
    X, y = regression_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_regression_models,
        task='regression'
    )

    predictions = blender._get_model_predictions(X)

    assert len(predictions) == len(fitted_regression_models)
    for pred in predictions:
        assert pred.shape == (len(X),)


def test_custom_parameters(fitted_classification_models):
    """Test custom initialization parameters"""
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3,
        max_iter=50,
    )

    assert blender.max_iter == 50


def test_weights_sum_to_one_multiple_models(fitted_classification_models, classification_data):
    """Test that weights always sum to 1 for multiple models"""
    X, y = classification_data
    blender = WeightedAverageBlender(
        fitted_models=fitted_classification_models,
        task='classification',
        n_classes=3
    )

    blender.fit(X, y)

    assert np.isclose(np.sum(blender.weights), 1.0, atol=1e-10)


def test_blend_predictions_binary_classification():
    """Test _blend_predictions for binary classification"""
    models = [{'model': Mock(), 'name': 'Model1'}]
    blender = WeightedAverageBlender(
        fitted_models=models,
        task='classification',
        n_classes=2
    )

    predictions = [np.array([[0.8], [0.3], [0.9]])]
    weights = np.array([1.0])

    result = blender._blend_predictions(predictions, weights, return_labels=True)

    expected = np.array([1, 0, 1])
    assert np.array_equal(result, expected)


def test_blend_predictions_without_labels():
    """Test _blend_predictions with return_labels=False"""
    models = [{'model': Mock(), 'name': 'Model1'}]
    blender = WeightedAverageBlender(
        fitted_models=models,
        task='classification',
        n_classes=3
    )

    pred1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    pred2 = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    predictions = [pred1, pred2]
    weights = np.array([0.6, 0.4])

    result = blender._blend_predictions(predictions, weights, return_labels=False)

    expected = 0.6 * pred1 + 0.4 * pred2
    assert np.allclose(result, expected)
