from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class StandardizeTransformer(TransformerMixin):
    def __init__(self, columns_to_standardize):
        self.columns_to_standardize = columns_to_standardize
        self.d = {}

    def fit(self, df=None, y=None):
        d = {}
        for feature_name in self.columns_to_standardize:
            mean = df[feature_name].mean()
            std = df[feature_name].std()
            d[feature_name] = (mean, std)
        self.d = d
        return self

    def transform(self, df):
        try:
            result = df.copy()
            for feature_name in self.columns_to_standardize:
                mean = self.d[feature_name][0]
                std = self.d[feature_name][1]
                result[feature_name] = (df[feature_name] - mean) / std
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
