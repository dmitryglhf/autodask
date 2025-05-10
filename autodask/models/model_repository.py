import json
import os
from typing import Dict, Tuple, Type, Any

from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import DaskLGBMClassifier, DaskLGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class AtomizedModel:
    """Repository of machine learning models with their hyperparameter spaces."""

    # Path to the model parameters JSON file
    PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'model_parameters.json')

    @classmethod
    def _load_parameter_spaces(cls) -> Dict[str, Dict[str, Any]]:
        """Load hyperparameter spaces from JSON file."""
        try:
            with open(cls.PARAMS_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            ValueError(f"Warning: Parameter space file not found at {cls.PARAMS_PATH}")

    @classmethod
    def get_classifier_models(cls) -> Dict[str, Tuple[Type, Dict[str, Any]]]:
        """Get classifier models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()

        models = {
            'l2_logreg': (LogisticRegression, param_spaces.get('l2_logreg', {})),
            # 'xgboost': (XGBClassifier, param_spaces.get('xgboost_clf', {})),
            # 'lgbm': (DaskLGBMClassifier, param_spaces.get('lgbm_clf', {})),
            # 'catboost': (CatBoostClassifier, param_spaces.get('catboost_clf', {}))
        }

        return models

    @classmethod
    def get_regressor_models(cls) -> Dict[str, Tuple[Type, Dict[str, Any]]]:
        """Get regressor models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()

        models = {
            'l2_linreg': (LinearRegression, param_spaces.get('l2_linreg', {})),
            # 'xgboost': (XGBRegressor, param_spaces.get('xgboost_reg', {})),
            # 'lgbm': (DaskLGBMRegressor, param_spaces.get('lgbm_reg', {})),
            # 'catboost': (CatBoostRegressor, param_spaces.get('catboost_reg', {}))
        }

        return models
