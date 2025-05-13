import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    General-purpose preprocessing class.

    Handles:
    - Missing value imputation
    - Feature scaling (numeric)
    - Categorical encoding (OneHot or Ordinal)

    Parameters
    ----------
    categorical_strategy : str, default='most_frequent'
        Strategy for imputing missing values in categorical columns.
        Options: 'most_frequent', 'constant', etc.

    numeric_strategy : str, default='mean'
        Strategy for imputing missing values in numeric columns.
        Options: 'mean', 'median', 'constant', etc.

    scaling : bool, default=True
        Whether to apply StandardScaler to numeric features.

    encoding : str, default='onehot'
        Encoding method for categorical features.
        Options:
            - 'onehot': OneHotEncoder
            - 'ordinal': OrdinalEncoder
            - 'none': no encoding
    """

    def __init__(self,
                 categorical_strategy='most_frequent',
                 numeric_strategy='mean',
                 scaling=True,
                 encoding='onehot'):
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy = numeric_strategy
        self.scaling = scaling
        self.encoding = encoding

        self.transformer = None
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with raw features.

        y : ignored
            Not used, present for API consistency.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Numeric preprocessing pipeline
        num_pipeline_steps = [('imputer', SimpleImputer(strategy=self.numeric_strategy))]
        if self.scaling:
            num_pipeline_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(steps=num_pipeline_steps)

        # Categorical preprocessing pipeline
        cat_pipeline = None
        if self.encoding != 'none' and self.categorical_features:
            cat_steps = [('imputer', SimpleImputer(strategy=self.categorical_strategy, fill_value='missing'))]

            if self.encoding == 'onehot':
                cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)))
            elif self.encoding == 'ordinal':
                cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")

            cat_pipeline = Pipeline(steps=cat_steps)

        # Combine numeric and categorical pipelines
        transformers = [('num', num_pipeline, self.numeric_features)]
        if cat_pipeline:
            transformers.append(('cat', cat_pipeline, self.categorical_features))

        self.transformer = ColumnTransformer(transformers=transformers)
        self.transformer.fit(X)

        return self

    def transform(self, X):
        """
        Apply transformations to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        if self.transformer is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        return self.transformer.transform(X)

    def get_feature_names(self):
        """
        Return feature names after transformation.

        Returns
        -------
        list of str
            List of transformed feature names.
        """
        feature_names = list(self.numeric_features)

        if self.encoding != 'none' and self.categorical_features:
            cat_pipeline = self.transformer.named_transformers_['cat']
            if self.encoding == 'onehot':
                encoder = cat_pipeline.named_steps['encoder']
                cat_names = encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_names)
            elif self.encoding == 'ordinal':
                feature_names.extend(self.categorical_features)

        return feature_names
