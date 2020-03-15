from sklearn.base import TransformerMixin


class NormalizeTransformer(TransformerMixin):
    def __init__(self, columns_to_normalize):
        self.columns_to_normalize = columns_to_normalize
        self.d = {}

    def fit(self, df=None, y=None):
        d = {}
        for feature_name in self.columns_to_normalize:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            d[feature_name] = (min_value, max_value)
        self.d = d
        return self

    def transform(self, df):
        try:
            result = df.copy()
            for feature_name in self.columns_to_normalize:
                min_value = self.d[feature_name][0]
                max_value = self.d[feature_name][1]
                if max_value != min_value:
                    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
                else:
                    pass
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
