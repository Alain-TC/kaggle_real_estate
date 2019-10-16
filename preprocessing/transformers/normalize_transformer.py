from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class NormalizeTransformer(TransformerMixin):
    def __init__(self, columns_to_fill):
        self.columns_to_fill = columns_to_fill


    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        try:
            result = df.copy()
            for feature_name in df.columns:
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)