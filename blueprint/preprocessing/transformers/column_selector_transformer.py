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
            cols_error = list(set(self.columns_to_keep) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class ExcludeColumnsTransformer(TransformerMixin):
    def __init__(self, columns_to_exclude):
        self.columns_to_exclude = columns_to_exclude

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):

        if all([c in df.columns for c in self.columns_to_exclude]):
            new_columns = [x for x in df.columns if x not in self.columns_to_exclude]
            return df[new_columns]
        else:
            cols_error = list(set(self.columns_to_exclude) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
