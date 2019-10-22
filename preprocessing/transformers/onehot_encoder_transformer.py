import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import numpy as np


class SimpleOneHotEncoder(TransformerMixin):
    def __init__(self, cols):
        """
        Transformer which executes one-hot encoding of given variables, without removing the other (not to be encoded)
        variables.
        :param cols: columns to one-hot encode
        """
        self.columns_to_encode = cols
        self.encoder = DictVectorizer(dtype=np.float64, separator='__')

    @staticmethod
    def _convert_columns_tostring(df):
        df = df.astype(str)
        return df

    def _split_columns(self, df):
        df_not_to_encode = df.drop(self.columns_to_encode, axis=1)
        df_to_encode = self._convert_columns_tostring(df[self.columns_to_encode])
        df_not_to_encode = df_not_to_encode.astype(np.float64)
        dict = df_to_encode.to_dict('records')
        return dict, df_not_to_encode

    def fit(self, df=None, y=None):
        dict_to_encode, df_not_to_encode = self._split_columns(df)
        self.encoder.fit(dict_to_encode)
        return self

    def transform(self, df):
        dict_to_encode, df_not_to_encode = self._split_columns(df)
        smat = self.encoder.transform(dict_to_encode)
        df_encoded = pd.DataFrame(smat.toarray(), columns=self.encoder.get_feature_names())
        df_encoded.index = df_not_to_encode.index
        df_tot = df_not_to_encode.merge(df_encoded, left_index=True, right_index=True, copy=False)
        df_tot.sort_index(axis=1, inplace=True)
        return df_tot
