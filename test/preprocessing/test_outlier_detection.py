# import unittest
# import numpy as np
# import pandas as pd
# from scipy.special import boxcox1p
# from test.context import blueprint
# from blueprint.preprocessing.transformers.box_cox_transformer import BoxCoxTransformer
#
#
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import IsolationForest
# from category_encoders import TargetEncoder
# from .transformers.fillna_transformer import FillnaMeanTransformer, FillnaMeanMatrixTransformer
# from .transformers.normalize_transformer import NormalizeTransformer
# from .transformers.add_column_transformer import CreateSumTransformer, CreateOneHotTransformer, NewHouseTransformer
# from .transformers.box_cox_transformer import BoxCoxTransformer
# from .transformers.column_selector_transformer import ExcludeColumnsTransformer
# from .transformers.onehot_encoder_transformer import SimpleOneHotEncoder
#
#
# def remove_outliers(data, columns_config):
#     quantitative_columns = columns_config["quantitative_columns"]
#     semi_quali_columns = columns_config["semi_quali_columns"]
#     qualitative_columns = columns_config["qualitative_columns"]
#     target_features_list = columns_config["target_features_list"]
#     indicator_features_list = columns_config["indicator_features_list"]
#
#     pipeline = make_pipeline(ExcludeColumnsTransformer(["Id"]),
#                              CreateSumTransformer(target_features_list),
#                              CreateOneHotTransformer(indicator_features_list),
#                              NewHouseTransformer(),
#                              BoxCoxTransformer(quantitative_columns),
#                              FillnaMeanTransformer(quantitative_columns),
#                              TargetEncoder(semi_quali_columns),
#                              SimpleOneHotEncoder(qualitative_columns),
#                              NormalizeTransformer(quantitative_columns),
#                              FillnaMeanMatrixTransformer()
#                              )
#
#     # Prepare Data Training
#     X = data
#     y = data[['SalePrice']]
#     X = pipeline.fit_transform(X, y)
#
#     # fit the model
#     clf = IsolationForest(max_samples=100)
#     clf.fit(X)
#     outlier_index = clf.predict(X)
#     clean_df = data[outlier_index == 1].reset_index(inplace=False, drop=True)
#
#     return clean_df
#
#
# class TestBoxCoxTransformer(unittest.TestCase):
#     def setUp(self):
#         column_c1 = np.array([0, 2, np.nan, 0, 2, 2, 0, 2, 0])
#         column_c2 = np.array([0, 1, np.nan, 2, 3, 4, 5.0, 6, np.nan])
#         column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
#         column_c4 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
#         self.df = pd.DataFrame({
#             "Id": column_c1,
#             "column_c2": column_c2,
#             "column_c3": column_c3,
#             "column_c4": column_c4
#         })
#         columns_config = {"quantitative_columns": [],
#          "semi_quali_columns": [],
#          "qualitative_columns": [],
#          "target_features_list": [],
#          "indicator_features_list": []
#          }
#
#         columns_to_transform = ["column_c1", "column_c3"]
#         self.boxCoxTransformer = BoxCoxTransformer(columns_to_transform)
#
#         lam = 0.15
#         column_c1 = boxcox1p(column_c1, lam)
#         column_c3 = boxcox1p(column_c3, lam)
#
#         self.filled_df = pd.DataFrame({
#             "column_c1": column_c1,
#             "column_c2": column_c2,
#             "column_c3": column_c3,
#             "column_c4": column_c4
#         })
#
#     def test_transform(self):
#         self.boxCoxTransformer.fit(self.df)
#         transformed_df = self.boxCoxTransformer.transform(self.df)
#         pd.testing.assert_frame_equal(transformed_df, self.filled_df)
