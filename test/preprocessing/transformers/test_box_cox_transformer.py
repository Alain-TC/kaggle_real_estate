import unittest
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from test.context import kaggle_blueprint
from collections import OrderedDict
from kaggle_blueprint.preprocessing.transformers.box_cox_transformer import BoxCoxTransformer


class TestBoxCoxTransformer(unittest.TestCase):
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
        columns_to_transform = ["column_c1", "column_c3"]
        self.boxCoxTransformer = BoxCoxTransformer(columns_to_transform)

        lam = 0.15
        column_c1 = boxcox1p(column_c1, lam)
        column_c3 = boxcox1p(column_c3, lam)

        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        )))

    def test_transform(self):
        self.boxCoxTransformer.fit(self.df)
        transformed_df = self.boxCoxTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
