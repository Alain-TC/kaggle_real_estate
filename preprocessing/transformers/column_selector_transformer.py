from sklearn.base import TransformerMixin


class KeepColumnsTransformer(TransformerMixin):
    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        try:
            return df[self.columns_to_keep]
        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
