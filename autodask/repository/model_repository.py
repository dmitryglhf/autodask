import json
import os
from typing import List, Dict, Any

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from autodask.core.data import ModelContainer


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
    def get_classifier_models(cls) -> List['ModelContainer']:
        """Get classifier models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()
        param_default = cls._load_parameter_default()

        models = [
            ModelContainer(
                model=LogisticRegression,
                model_name='l2_logreg',
                model_task_type='classification',
                hyperparameters=param_default.get('l2_logreg', {}),
                search_space=param_spaces.get('l2_logreg', {})
            ),
            ModelContainer(
                model=RandomForestClassifier,
                model_name='random_forest',
                model_task_type='classification',
                hyperparameters=param_default.get('rf_clf', {}),
                search_space=param_spaces.get('rf_clf', {})
            ),
            ModelContainer(
                model=LGBMClassifier,
                model_name='lgbm',
                model_task_type='classification',
                hyperparameters=param_default.get('lgbm_clf', {}),
                search_space=param_spaces.get('lgbm_clf', {})
            ),
            ModelContainer(
                model=XGBClassifier,
                model_name='xgboost',
                model_task_type='classification',
                hyperparameters=param_default.get('xgboost_clf', {}),
                search_space=param_spaces.get('xgboost_clf', {})
            ),
            ModelContainer(
                model=CatBoostClassifier,
                model_name='catboost',
                model_task_type='classification',
                hyperparameters=param_default.get('catboost_clf', {}),
                search_space=param_spaces.get('catboost_clf', {})
            )
        ]

        return models

    @classmethod
    def get_regressor_models(cls) -> List['ModelContainer']:
        """Get regressor models with their hyperparameter spaces."""
        param_spaces = cls._load_parameter_spaces()
        param_default = cls._load_parameter_default()

        models = [
            ModelContainer(
                model=LinearRegression,
                model_name='l2_linreg',
                model_task_type='regression',
                hyperparameters=param_default.get('l2_linreg', {}),
                search_space=param_spaces.get('rf_reg', {})
            ),
            ModelContainer(
                model=RandomForestRegressor,
                model_name='random_forest',
                model_task_type='regression',
                hyperparameters=param_default.get('rf_reg', {}),
                search_space=param_spaces.get('rf_reg', {})
            ),
            ModelContainer(
                model=LGBMRegressor,
                model_name='lgbm',
                model_task_type='regression',
                hyperparameters=param_default.get('lgbm_reg', {}),
                search_space=param_spaces.get('lgbm_reg', {})
            ),
            ModelContainer(
                model=XGBRegressor,
                model_name='xgboost',
                model_task_type='regression',
                hyperparameters=param_default.get('xgboost_reg', {}),
                search_space=param_spaces.get('xgboost_reg', {})
            ),
            ModelContainer(
                model=CatBoostRegressor,
                model_name='catboost',
                model_task_type='regression',
                hyperparameters=param_default.get('catboost_reg', {}),
                search_space=param_spaces.get('catboost_reg', {})
            )
        ]

        return models
