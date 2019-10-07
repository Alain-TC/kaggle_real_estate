from sklearn.base import TransformerMixin, BaseEstimator


class DataframeToMatrix(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df.to_numpy()