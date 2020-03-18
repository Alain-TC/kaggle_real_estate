from sklearn.base import TransformerMixin


class CreateSumTransformer(TransformerMixin):
    def __init__(self, target_features_list):
        self.target_features_list = target_features_list

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        for couple in self.target_features_list:
            self.target_name = couple[0]
            self.feature_list = couple[1]
            df[self.target_name] = sum([df[column_name] for column_name in self.feature_list])
            df[self.target_name] = df[self.target_name].fillna(0)
        return df


class CreateOneHotTransformer(TransformerMixin):
    def __init__(self, features_list):
        self.features_list = features_list

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        for feature in self.features_list:
            df["Has{}".format(feature)] = df[feature].apply(lambda x: 1 if x > 0 else 0)
        return df


class NewHouseTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        df['NewHouse'] = 0
        idx = df[df['YrSold'] == df['YearBuilt']].index
        df.loc[idx, 'NewHouse'] = 1
        return df
