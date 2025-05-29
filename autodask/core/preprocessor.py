import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
                                   OrdinalEncoder,
                                   LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from autodask.utils.log import get_logger


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categorical_strategy: str = 'most_frequent',
                 numeric_strategy: str = 'mean',
                 scaling: bool = True,
                 encoding: str = 'onehot',
                 target_encoding: bool = True):
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy = numeric_strategy
        self.scaling = scaling
        self.encoding = encoding
        self.target_encoding = target_encoding

        self.transformer_: ColumnTransformer | None = None
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.target_encoder_: LabelEncoder | None = None

        self.log = get_logger(self.__class__.__name__)

    def fit(self, X, y=None):
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(
            include=['object', 'category', 'bool']).columns.tolist()

        # numeric pipeline
        num_steps = [('imputer', SimpleImputer(strategy=self.numeric_strategy))]
        if self.scaling:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)

        # categorical pipeline
        cat_pipeline = None
        if self.encoding != 'none' and self.categorical_features_:
            cat_steps = [('imputer', SimpleImputer(strategy=self.categorical_strategy,
                                                   fill_value='missing'))]

            if self.encoding == 'onehot':
                cat_steps.append(('encoder',
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            elif self.encoding == 'ordinal':
                cat_steps.append(('encoder',
                                  OrdinalEncoder(handle_unknown='use_encoded_value',
                                                 unknown_value=-1)))
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")

            cat_pipeline = Pipeline(steps=cat_steps)

        # merge
        transformers = [('num', num_pipeline, self.numeric_features_)]
        if cat_pipeline:
            transformers.append(('cat', cat_pipeline, self.categorical_features_))

        self.transformer_ = ColumnTransformer(transformers=transformers)
        self.transformer_.fit(X)

        # target encoder (if handed)
        if y is not None and self.target_encoding:
            y_series = pd.Series(y)
            if (y_series.dtype.kind not in 'ifc') or y_series.dtype == bool:
                # strings / categories / bool --> LabelEncoder
                self.target_encoder_ = LabelEncoder()
                self.target_encoder_.fit(y_series)
                self.log.info('Target encoder fitted with classes: %s',
                              list(self.target_encoder_.classes_))
            else:
                # except of numeric target
                self.target_encoder_ = None

        return self

    def transform(self, X):
        if self.transformer_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        return self.transformer_.transform(X)

    def transform_target(self, y):
        if self.target_encoder_ is None or y is None:
            return y
        return self.target_encoder_.transform(y)

    def decode_target(self, y_encoded):
        if self.target_encoder_ is None:
            return y_encoded
        return self.target_encoder_.inverse_transform(y_encoded)

    def fit_transform(self, X, y=None, **fit_params):
        out = self.fit(X, y).transform(X)
        if y is not None:
            y_tr = self.transform_target(y)
            return out, y_tr
        return out

    def get_feature_names(self) -> list[str]:
        feature_names = list(self.numeric_features_)

        if self.encoding != 'none' and self.categorical_features_:
            cat_pipeline = self.transformer_.named_transformers_['cat']
            if self.encoding == 'onehot':
                encoder = cat_pipeline.named_steps['encoder']
                cat_names = encoder.get_feature_names_out(self.categorical_features_)
                feature_names.extend(cat_names)
            elif self.encoding == 'ordinal':
                feature_names.extend(self.categorical_features_)

        return feature_names
