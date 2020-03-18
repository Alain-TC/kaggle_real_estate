import collections
import unittest
import numpy as np
import pandas as pd
from test.context import kaggle_blueprint
from kaggle_blueprint.preprocessing.transformers.log_target_transformer import transform_log, transform_exp


class Testtransform_log_exp(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, np.nan, 2, 3, 4, 5.0, 6, np.nan])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame(collections.OrderedDict([
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        ]))
        self.log_column_name = "column_c2"
        self.exp_column_name = "column_c3"

        column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([-np.inf, 0.0, np.nan, np.log(2), np.log(3), np.log(4), np.log(5.0), np.log(6), np.nan])
        column_c3 = np.array([1., np.exp(1), 1., 1., 1., 1., np.exp(1), 1., np.nan])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.filled_df = pd.DataFrame(collections.OrderedDict([
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        ]))

    def test_transform(self):
        transformed_df = transform_log(self.df, self.log_column_name)
        transformed_df = transform_exp(transformed_df, self.exp_column_name)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
