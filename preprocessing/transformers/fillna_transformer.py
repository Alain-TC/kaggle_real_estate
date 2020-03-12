from sklearn.base import TransformerMixin
import numpy as np


class FillnaMeanTransformer(TransformerMixin):
    def __init__(self, columns_to_fill):
        self.columns_to_fill = columns_to_fill

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()
        try:
            for column in self.columns_to_fill:
                df_temp = df_copy[[column]]
                column_copy = df_copy[[column]]
                column_copy = column_copy.dropna()
                moyenne_colonne = np.mean(column_copy)

                if np.isnan(float(moyenne_colonne)):
                    values = {column: float(0)}
                else:
                    values = {column: float(moyenne_colonne)}
                df_temp.fillna(value=values, inplace=True)
                df[[column]] = df_temp
            return df

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class FillnaMeanMatrixTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        where_are_NaNs = np.isnan(df)
        df[where_are_NaNs] = 0
        return df
