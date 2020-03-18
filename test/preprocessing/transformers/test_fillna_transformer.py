import unittest
import numpy as np
import pandas as pd
from test.context import kaggle_blueprint
from collections import OrderedDict
from kaggle_blueprint.preprocessing.transformers.fillna_transformer import FillnaMeanTransformer


class TestFillnaMeanTransformer(unittest.TestCase):
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
        columns_to_fill = ["column_c1", "column_c2", "column_c3"]
        self.fillnaMeanTransformer = FillnaMeanTransformer(columns_to_fill)

        column_c1 = np.array([0.0, 2, 1, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, 3, 2, 3, 4, 5.0, 6, 3])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0.25])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        )))

    def test_transform(self):
        self.fillnaMeanTransformer.fit(self.df)
        transformed_df = self.fillnaMeanTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
