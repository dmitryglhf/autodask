from enum import Enum

from dask_ml.linear_model import LinearRegression, LogisticRegression
from dask_ml.xgboost import XGBClassifier, XGBRegressor
from lightgbm import DaskLGBMClassifier, DaskLGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class AtomizedModel(Enum):
    CLF_MODELS = dict(
        lr=LogisticRegression,
        xgb=XGBClassifier,
        lgbm=DaskLGBMClassifier,
        cb=CatBoostClassifier
    )

    REG_MODELS = dict(
        lr=LinearRegression,
        xgb=XGBRegressor,
        lgbm=DaskLGBMRegressor,
        cb=CatBoostRegressor
    )
