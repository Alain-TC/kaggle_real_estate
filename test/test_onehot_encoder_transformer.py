import collections
import unittest
import numpy as np
import pandas as pd
from .context import blueprint
from blueprint.preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder


class TestOneHotTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array(["male", "female", "female", "male", "male", "male", "female", "female", "male"])
        column_c2 = np.array(["France", "Spain", "Spain", "Germany", "Italy", "France", "Italy", "Germany", "Spain"])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame({
            "sex": column_c1,
            "country": column_c2,
            "column_c3": column_c3
        })
        self.columns_to_onehot = ["sex", "country"]
        self.onehotTransformer = SimpleOneHotEncoder(self.columns_to_onehot)

        male = np.array([1., 0., 0., 1., 1., 1., 0., 0., 1.])
        female = np.array([0., 1., 1., 0., 0., 0, 1., 1., 0.])
        france = np.array([1., 0., 0., 0., 0., 1, 0., 0., 0.])
        spain = np.array([0., 1., 1., 0., 0., 0, 0., 0., 1.])
        germany = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.])
        italy = np.array([0., 0., 0., 0., 1., 0., 1, 0., 0.])
        self.filled_df = pd.DataFrame(collections.OrderedDict([
            ("column_c3", column_c3),
            ("country__France", france),
            ("country__Germany", germany),
            ("country__Italy", italy),
            ("country__Spain", spain),
            ("sex__female", female),
            ("sex__male", male)
        ]))

    def test_transform(self):
        self.onehotTransformer.fit(self.df)
        transformed_df = self.onehotTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
