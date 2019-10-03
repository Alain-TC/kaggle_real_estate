from sklearn.base import TransformerMixin, BaseEstimator


class DataframeToMatrix(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        print("coucou")
        print(df)
        return df.to_numpy()