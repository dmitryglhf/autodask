from enum import Enum

from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import DaskLGBMClassifier, DaskLGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor


class AtomizedModel(Enum):
    CLF_MODELS = dict(
        l2_logreg=LogisticRegression(),
        # xgb=XGBClassifier(),
        lgbm=DaskLGBMClassifier(verbose=-1, n_jobs=1),
        # cb=CatBoostClassifier()
    )

    REG_MODELS = dict(
        l2_linreg=LinearRegression(),
        # xgb=XGBRegressor(),
        lgbm=DaskLGBMRegressor(verbose=-1, n_jobs=1),
        # cb=CatBoostRegressor()
    )
