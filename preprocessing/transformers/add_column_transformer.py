from sklearn.base import TransformerMixin


class CreateTotalSFTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        return df