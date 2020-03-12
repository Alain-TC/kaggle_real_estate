def get_config_columns():
    columns_categories = {"quantitative_columns": ["MSSubClass", "LotFrontage", "OverallQual", "OverallCond",
                                                   "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF2", "BsmtUnfSF",
                                                   "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "BedroomAbvGr",
                                                   "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
                                                   "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                                                   "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
                                                   "MoSold", "YrSold", "LotArea", "BsmtFinSF1"],
                          "qualitative_columns": ['MSZoning', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood',
                                                  'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                                                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                                                  'Heating', 'Electrical', 'Functional', 'GarageType', 'MiscFeature',
                                                  'SaleType', 'SaleCondition'],
                          "semi_quali_columns": ['Street', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual',
                                                 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                                 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
                                                 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                                                 'PavedDrive', 'PoolQC', 'Fence']
                          }
    return columns_categories
