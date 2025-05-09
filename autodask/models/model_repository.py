from enum import Enum

from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import DaskLGBMClassifier, DaskLGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor


# изменить структуру на: name, (model_class, param_space)
# добавить json с гиперпараметрами моделей
class AtomizedModel(Enum):
    CLF_MODELS = dict(
        l2_logreg=LogisticRegression(**params),
        # xgboost=XGBClassifier(),
        lgbm=DaskLGBMClassifier(**params),
        # catboost=CatBoostClassifier()
    )

    REG_MODELS = dict(
        l2_linreg=LinearRegression(**params),
        # xgboost=XGBRegressor(),
        lgbm=DaskLGBMRegressor(**params),
        # catboost=CatBoostRegressor()
    )
