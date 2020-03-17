import unittest
from collections import OrderedDict
from test.context import blueprint
import pandas as pd
import numpy as np
from blueprint.preprocessing.split_dataframe import split_dataframe_by_row


class Testsplit_dataframe_by_row(unittest.TestCase):
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

        split_1_column_c1 = np.array([2., 2.])
        split_1_column_c2 = np.array([6., 1.])
        split_1_column_c3 = np.array([0., 1.])
        split_1_column_c4 = np.array([0., 1.])

        split_2_column_c1 = np.array([2., 0., 0., np.nan, 2., 0., 0.])
        split_2_column_c2 = np.array([4., 0., np.nan, np.nan, 3., 2., 5.])
        split_2_column_c3 = np.array([0., 0., np.nan, 0., 0., 0., 1.])
        split_2_column_c4 = np.array([0., 0., np.nan, 0., 0., 0., 1.])

        print(split_dataframe_by_row(self.df, .5))
        self.split_df_1 = pd.DataFrame({
            "column_c1": split_1_column_c1,
            "column_c2": split_1_column_c2,
            "column_c3": split_1_column_c3,
            "column_c4": split_1_column_c4
        })

        self.split_df_2 = pd.DataFrame({
            "column_c1": split_2_column_c1,
            "column_c2": split_2_column_c2,
            "column_c3": split_2_column_c3,
            "column_c4": split_2_column_c4
        })

    def test_transform(self):
        transformed_df_1, transformed_df_2 = split_dataframe_by_row(self.df, .25)
        pd.testing.assert_frame_equal(transformed_df_1, self.split_df_1)
        pd.testing.assert_frame_equal(transformed_df_2, self.split_df_2)
