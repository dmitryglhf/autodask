from enum import Enum

from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import DaskLGBMClassifier, DaskLGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class AtomizedModel(Enum):
    CLF_MODELS = dict(
        l2_logreg=LogisticRegression,
        xgb=XGBClassifier,
        lgbm=DaskLGBMClassifier,
        cb=CatBoostClassifier
    )

    REG_MODELS = dict(
        l2_linreg=Ridge,
        xgb=XGBRegressor,
        lgbm=DaskLGBMRegressor,
        cb=CatBoostRegressor
    )
