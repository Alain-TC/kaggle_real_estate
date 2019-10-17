import unittest
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
import pandas as pd
import numpy as np


class TestNormalizeTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, 1, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, 3, 2, 3, 4, 5.0, 1, 2])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 2])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame({
            "column_c1": column_c1,
            "column_c2": column_c2,
            "column_c3": column_c3,
            "column_c4": column_c4
        })
        self.columns_to_fill = ["column_c1", "column_c2", "column_c3"]
        self.normalizeTransformer = NormalizeTransformer(self.columns_to_fill)


        column_c1 = np.array([0.0, 1, 0.5, 0, 1, 1, 0, 1, 0])
        column_c2 = np.array([0, 0.2, 0.6, 0.4, 0.6, 0.8, 1, 0.2, 0.4])
        column_c3 = np.array([0, 0.5, 0, 0, 0, 0, 0.5, 0, 1])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.filled_df = pd.DataFrame({
            "column_c1": column_c1,
            "column_c2": column_c2,
            "column_c3": column_c3,
            "column_c4": column_c4
        })


    def test_fit(self):
        self.normalizeTransformer.fit(self.df)

        for column in self.columns_to_fill:
            self.assertEqual(self.normalizeTransformer.d[column], (self.df[column].min(),self.df[column].max()))


    def test_transform(self):
        self.normalizeTransformer.fit(self.df)
        transformed_df = self.normalizeTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
