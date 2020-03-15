import collections
import unittest

import numpy as np
import pandas as pd

from blueprint.preprocessing.transformers import StandardizeTransformer


class TestStandardizeTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, 1, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 0, 3, 1, 2, 4, 4, 1, 2])
        column_c3 = np.array([0, 1, 0, 5, 1, 0, 1, 0, 1])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame({
            "column_c1": column_c1,
            "column_c2": column_c2,
            "column_c3": column_c3,
            "column_c4": column_c4
        })
        self.columns_to_standardize = ["column_c1", "column_c2", "column_c3"]
        self.standardizeTransformer = StandardizeTransformer(self.columns_to_standardize)

        column_c1 = np.array([-1, 1, 0, -1.0, 1, 1, -1, 1, -1])
        column_c2 = np.array([-1.2292725943057183, -1.2292725943057183, 0.7231015260621872, -0.5784812208497497,
                              0.07231015260621876, 1.3738928995181559, 1.3738928995181559, -0.5784812208497497,
                              0.07231015260621876])
        column_c3 = np.array([-0.6324555320336759, 0.0, -0.6324555320336759, 2.5298221281347035, 0.0,
                              -0.6324555320336759, 0.0, -0.6324555320336759, 0.0])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.filled_df = pd.DataFrame(collections.OrderedDict([
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        ]))

    def test_fit(self):
        self.standardizeTransformer.fit(self.df)

        for column in self.columns_to_standardize:
            self.assertEqual(self.standardizeTransformer.d[column], (self.df[column].mean(), self.df[column].std()))

    def test_transform(self):
        self.standardizeTransformer.fit(self.df)
        transformed_df = self.standardizeTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
