import numpy as np

from autodask.utils.log import get_logger
from sklearn.model_selection import KFold, StratifiedKFold

from autodask.utils.regular_functions import is_classification_task


def evaluate_model(model_class, params, X, y, metric_func, task: str, cv_folds=5):
    """Evaluate a parameter set using k-fold cross-validation.

    Args:
        model_class: Model class to instantiate
        params (dict): Parameters for model initialization
        X: Features dataset
        y: Target values
        metric_func (callable): Scoring function
        task (str): Task, supported 'classification' and 'regression'
        cv_folds (int): Number of folds for cross-validation

    Returns:
        float: Mean evaluation score across all folds

    Note:
        Returns -inf for invalid parameter sets to handle optimization failures
    """
    log = get_logger('CrossValidation')
    try:
        # Initialize K-fold cross-validator
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_classification_task(task) \
            else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scores = []

        # Perform k-fold cross-validation
        for train_index, val_index in kf.split(X, y if is_classification_task(task) else None):
            X_train, X_val = X.iloc[train_index] if hasattr(X, 'iloc') else X[train_index], X.iloc[
                val_index] if hasattr(X, 'iloc') else X[val_index]
            y_train, y_val = y.iloc[train_index] if hasattr(y, 'iloc') else y[train_index], y.iloc[
                val_index] if hasattr(y, 'iloc') else y[val_index]

            y_val = np.ravel(y_val)

            # Train and evaluate model on this fold
            model = model_class(**params)
            model.fit(X_train, y_train)
            if is_classification_task(task):
                y_pred = model.predict_proba(X_val)
            else:
                y_pred = model.predict(X_val)
            fold_score = metric_func(y_val, y_pred)
            scores.append(fold_score)

        # Return mean score across all folds
        mean_score = sum(scores) / len(scores)
        log.debug(f"Mean {cv_folds}-fold CV score for params {params}: {mean_score}")
        return mean_score

    except Exception as e:
        raise ValueError("Invalid parameters for evaluating")
