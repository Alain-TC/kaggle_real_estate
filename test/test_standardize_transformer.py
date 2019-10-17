import unittest
from preprocessing.transformers.standardize_transformer import StandardizeTransformer
import pandas as pd
import numpy as np


class TestStandardizeTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, 1, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, 3, 1, 2, 4, 4, 1, 2])
        column_c3 = np.array([0, 1, 0, 5, 1, 0, 1, 0, 1])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame({
            "column_c1": column_c1,
            "column_c2": column_c2,
            "column_c3": column_c3,
            "column_c4": column_c4
        })
        columns_to_fill = ["column_c1", "column_c2", "column_c3"]
        self.standardizeTransformer = StandardizeTransformer(columns_to_fill)


        column_c1 = np.array([-1, 1, 0, -1.0, 1, 1, -1, 1, -1])
        column_c2 = np.array([-1, -0.5, 0.5, -0.5, 0, 1, 1, -0.5, 0])
        column_c3 = np.array([-0.25, 0, -0.25, 1, 0, -0.25, 0, -0.25, 0])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.filled_df = pd.DataFrame({
            "column_c1": column_c1,
            "column_c2": column_c2,
            "column_c3": column_c3,
            "column_c4": column_c4
        })

    def test_transform(self):
        self.standardizeTransformer.fit(self.df)
        transformed_df = self.standardizeTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
