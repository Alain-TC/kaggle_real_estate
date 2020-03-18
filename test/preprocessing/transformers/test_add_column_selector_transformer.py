import unittest
import numpy as np
import pandas as pd
from collections import OrderedDict
from test.context import kaggle_blueprint
from kaggle_blueprint.preprocessing.transformers.add_column_transformer import CreateSumTransformer, CreateOneHotTransformer, \
    NewHouseTransformer


class TestNewHouseTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array([0, 1., np.nan, 0, 2, 2, 0, 2, 0])
        column_c2 = np.array([0, 1, np.nan, 2, 3, 4, 5.0, 6, np.nan])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame(OrderedDict((
            ("YrSold", column_c1),
            ("YearBuilt", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4)
        )))
        self.newHouseTransformer = NewHouseTransformer()

        NewHouse = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])

        self.filled_df = pd.DataFrame(OrderedDict((
            ("YrSold", column_c1),
            ("YearBuilt", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4),
            ("NewHouse", NewHouse)
        )))

    def test_transform(self):
        self.newHouseTransformer.fit(self.df)
        transformed_df = self.newHouseTransformer.transform(self.df)
        print(transformed_df)

        print(self.filled_df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)


class TestCreateOneHotTransformer(unittest.TestCase):
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
        features_list = ["column_c1", "column_c2"]
        self.createOneHotTransformer = CreateOneHotTransformer(features_list)

        Hascolumn_c1 = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
        Hascolumn_c2 = np.array([0, 1, 0, 1, 1, 1, 1, 1, 0])

        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4),
            ("Hascolumn_c1", Hascolumn_c1),
            ("Hascolumn_c2", Hascolumn_c2)
        )))

    def test_transform(self):
        self.createOneHotTransformer.fit(self.df)
        transformed_df = self.createOneHotTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)


class TestCreateSumTransformer(unittest.TestCase):
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
        target_features_list = [["column_c5", ["column_c3", "column_c4"]], ["column_c6", ["column_c1", "column_c4"]]]
        self.createSumTransformer = CreateSumTransformer(target_features_list)

        column_c5 = np.array([0., 2., 0., 0., 0., 0., 2., 0., 0.])
        column_c6 = np.array([0., 3., 0., 0., 2., 2., 1., 2., 0.])

        self.filled_df = pd.DataFrame(OrderedDict((
            ("column_c1", column_c1),
            ("column_c2", column_c2),
            ("column_c3", column_c3),
            ("column_c4", column_c4),
            ("column_c5", column_c5),
            ("column_c6", column_c6)
        )))

    def test_transform(self):
        self.createSumTransformer.fit(self.df)
        transformed_df = self.createSumTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
