from sklearn.base import TransformerMixin


class CreateTotalSFTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        return df


class NewFeaturesTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df['YearBuiltRemod'] = df['YearBuilt'] + df['YearRemodAdd']
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalSquareFootage'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df[
            'WoodDeckSF']
        df['OverallRating'] = df['OverallQual'] + df['OverallCond']

        df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        df['NewHouse'] = 0
        idx = df[df['YrSold'] == df['YearBuilt']].index
        df.loc[idx, 'NewHouse'] = 1
        return df
