import json
import os
from typing import Dict, Tuple, Type, Any

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class AtomizedModel:
    """Repository of ml models with their default hyperparameters and hyperparameters spaces."""

    # Path to the model parameters JSON file
    SEARCH_PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'search_parameters.json')
    DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'default_parameters.json')

    @classmethod
    def _load_parameter_spaces(cls) -> Dict[str, Dict[str, Any]]:
        """Load hyperparameters spaces from JSON file."""
        try:
            with open(cls.SEARCH_PARAMS_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            ValueError(f"Parameter space file not found at {cls.SEARCH_PARAMS_PATH}")

    @classmethod
    def _load_parameter_default(cls) -> Dict[str, Dict[str, Any]]:
        """Load default hyperparameters from JSON file."""
        try:
            with open(cls.DEFAULT_PARAMS_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            ValueError(f"Parameter space file not found at {cls.DEFAULT_PARAMS_PATH}")

    @classmethod
    def get_classifier_models(cls) -> Dict[str, Tuple[Type, Dict[str, Any], Dict[str, Any]]]:
        """Get classifier models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()
        param_default = cls._load_parameter_default()

        models = {
            'l2_logreg': (
                LogisticRegression,
                param_spaces.get('l2_logreg', {}),
                param_default.get('l2_logreg', {}),
            ),
            'random_forest': (
                RandomForestClassifier,
                param_spaces.get('rf_clf', {}),
                param_default.get('rf_clf', {}),
            ),
            'lgbm': (
                LGBMClassifier,
                param_spaces.get('lgbm_clf', {}),
                param_default.get('lgbm_clf', {})
            ),
            'xgboost': (
                XGBClassifier,
                param_spaces.get('xgboost_clf', {}),
                param_default.get('xgboost_clf', {})
            ),
            'catboost': (
                CatBoostClassifier,
                param_spaces.get('catboost_clf', {}),
                param_default.get('catboost_clf', {})
            ),
        }

        return models

    @classmethod
    def get_regressor_models(cls) -> Dict[str, Tuple[Type, Dict[str, Any], Dict[str, Any]]]:
        """Get regressor models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()
        param_default = cls._load_parameter_default()

        models = {
            'l2_linreg': (
                LinearRegression,
                param_spaces.get('l2_linreg', {}),
                param_default.get('l2_linreg', {}),
            ),
            'random_forest': (
                RandomForestRegressor,
                param_spaces.get('rf_reg', {}),
                param_default.get('rf_reg', {}),
            ),
            'lgbm': (
                LGBMRegressor,
                param_spaces.get('lgbm_reg', {}),
                param_default.get('lgbm_reg', {})
            ),
            'xgboost': (
                XGBRegressor,
                param_spaces.get('xgboost_reg', {}),
                param_default.get('xgboost_reg', {})
            ),
            'catboost': (
                CatBoostRegressor,
                param_spaces.get('catboost_reg', {}),
                param_default.get('catboost_reg', {})
            ),
        }

        return models
