from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class FillnaTransformer(TransformerMixin):
    def __init__(self, columns_to_fill):
        self.columns_to_fill = columns_to_fill


    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()
        try:
            for column in self.columns_to_fill:
                column_copy = df_copy[[column]]
                column_copy = column_copy.dropna()
                moyenne_colonne = np.mean(column_copy)
                values = {column: moyenne_colonne}
                df.fillna(value=values)
            return df

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)