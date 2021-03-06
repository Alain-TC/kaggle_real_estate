import unittest
import numpy as np
import pandas as pd
from test.context import kaggle_blueprint
from collections import OrderedDict
from kaggle_blueprint.preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer, \
    ExcludeColumnsTransformer


class TestKeepColumnsTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, np.nan, 2, 3, 4, 5.0, 6, np.nan])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        )))
        columns_to_keep = ["column_c1", "column_c3"]
        self.keepColumnsTransformer = KeepColumnsTransformer(columns_to_keep)

        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])

        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c3", column_c3)
        )))

    def test_transform(self):
        self.keepColumnsTransformer.fit(self.df)
        transformed_df = self.keepColumnsTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)


class TestExcludeColumnsTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, np.nan, 2, 3, 4, 5.0, 6, np.nan])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        )))
        columns_to_exclude = ["column_c2", "column_c4"]
        self.excludeColumnsTransformer = ExcludeColumnsTransformer(columns_to_exclude)

        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])

        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c3", column_c3)
        )))

    def test_transform(self):
        self.excludeColumnsTransformer.fit(self.df)
        transformed_df = self.excludeColumnsTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
