from sklearn.base import TransformerMixin
from scipy.special import boxcox1p


class BoxCoxTransformer(TransformerMixin):
    def __init__(self, columns_to_transform=None):
        self.columns_to_transform = columns_to_transform

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()
        skewed_features = self.columns_to_transform
        lam = 0.15
        for feat in skewed_features:
            # all_data[feat] += 1
            df_copy[feat] = boxcox1p(df_copy[feat], lam)

        return df_copy
