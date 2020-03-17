import unittest
from blueprint.modelisation.config_columns import get_config_columns


class Testget_config_columns(unittest.TestCase):
    def setUp(self):
        self.key_set = {"quantitative_columns", "semi_quali_columns", "qualitative_columns", "target_features_list",
                        "indicator_features_list"}

        self.quantitative_columns = ["MSSubClass", "LotFrontage", "OverallQual", "OverallCond",
                                     "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF2", "BsmtUnfSF",
                                     "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "BedroomAbvGr",
                                     "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
                                     "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                                     "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
                                     "MoSold", "YrSold", "LotArea", "BsmtFinSF1"]
        self.qualitative_columns = ['MSZoning', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood',
                                    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                                    'Heating', 'Electrical', 'Functional', 'GarageType', 'MiscFeature',
                                    'SaleType', 'SaleCondition']
        self.semi_quali_columns = ['Street', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual',
                                   'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                   'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
                                   'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                                   'PavedDrive', 'PoolQC', 'Fence']
        self.target_features_list = [['TotalSF', ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']],
                                     ["YearBuiltRemod", ["YearBuilt", "YearRemodAdd"]],
                                     ["TotalSF", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]],
                                     ["TotalSquareFootage",
                                      ["BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF"]],
                                     ["TotalBath",
                                      ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]],
                                     ["TotalPorchSF",
                                      ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch",
                                       "WoodDeckSF"]], ["OverallRating", ["OverallQual", "OverallCond"]]]
        self.indicator_features_list = ["PoolArea", "2ndFlrSF", "GarageArea", "TotalBsmtSF", "Fireplaces"]

    def test_transform(self):
        columns_config = get_config_columns()

        self.assertEqual(columns_config.keys(), self.key_set)
        self.assertListEqual(self.quantitative_columns, columns_config["quantitative_columns"])
        self.assertListEqual(self.semi_quali_columns, columns_config["semi_quali_columns"])
        self.assertListEqual(self.qualitative_columns, columns_config["qualitative_columns"])
        self.assertListEqual(self.target_features_list, columns_config["target_features_list"])
        self.assertListEqual(self.indicator_features_list, columns_config["indicator_features_list"])
