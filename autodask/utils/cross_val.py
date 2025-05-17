from autodask.utils.log import get_logger
from sklearn.model_selection import KFold


def evaluate_model(model_class, params, X, y, metric_func, cv_folds=5):
    """Evaluate a parameter set using k-fold cross-validation.

    Args:
        model_class: Model class to instantiate
        params (dict): Parameters for model initialization
        X: Features dataset
        y: Target values
        metric_func (callable): Scoring function
        n_splits (int): Number of folds for cross-validation

    Returns:
        float: Mean evaluation score across all folds

    Note:
        Returns -inf for invalid parameter sets to handle optimization failures
    """
    log = get_logger('CrossValidation')
    try:
        # Initialize K-fold cross-validator
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scores = []

        # Perform k-fold cross-validation
        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index] if hasattr(X, 'iloc') else X[train_index], X.iloc[
                val_index] if hasattr(X, 'iloc') else X[val_index]
            y_train, y_val = y.iloc[train_index] if hasattr(y, 'iloc') else y[train_index], y.iloc[
                val_index] if hasattr(y, 'iloc') else y[val_index]

            # Train and evaluate model on this fold
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_score = metric_func(y_val, y_pred)
            scores.append(fold_score)

        # Return mean score across all folds
        mean_score = sum(scores) / len(scores)
        log.debug(f"Mean {cv_folds}-fold CV score for params {params}: {mean_score}")
        return mean_score

    except Exception as e:
        log.debug(f"Error evaluating params {params}: {e}")
        return float('-inf')  # Return a very bad score for invalid parameters
