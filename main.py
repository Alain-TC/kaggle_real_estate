import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix
from preprocessing.transformers.log_target_transformer import transform_log, transform_exp
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression
from preprocessing.split_dataframe import split_dataframe_by_row
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp

from preprocessing.transformers.fillna_transformer import FillnaMeanTransformer
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
from modelisation.model import FullModelClass
import warnings
from preprocessing.transformers.standardize_transformer import StandardizeTransformer
from preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder
warnings.filterwarnings('ignore')
#from preprocessing.transformers.normalize_transformer import NormalizeTransformer


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv("{}/data/train.csv".format(dir_path))


    # Transformation log(target)
    df_train = transform_log(df_train, 'SalePrice')

    # split Train/Eval
    df_train, df_train_eval = split_dataframe_by_row(df_train, 0.7)

    # Preprocess data
    quantitative_columns =["MSSubClass","LotFrontage","OverallQual","OverallCond","YearBuilt","YearRemodAdd",
                           "MasVnrArea","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF",
                           "GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr",
                           "KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea",
                            "WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal",
                           "MoSold","YrSold","LotArea"]

    qualitative_columns = ['Id', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour','Utilities', 'LotConfig',
                           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
                           'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                           'BsmtFinSF1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                           'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'PoolQC','Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


    # Pipeline
    processing_pipeline = make_pipeline(SimpleOneHotEncoder(qualitative_columns),
                                        KeepColumnsTransformer(quantitative_columns),
                                        FillnaMeanTransformer(quantitative_columns),
                                        # NormalizeTransformer(quantitative_columns)
                                        StandardizeTransformer(quantitative_columns),
                                        NormalizeTransformer(quantitative_columns),
                                        SelectKBest(score_func=f_regression, k=36)
                                        )


    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]

    hyperopt = True
    model_type = "Linear"

    # Model
    if model_type == "Lasso":
        model = linear_model.LinearRegression()
        space = {
                 "selectkbest__k": (1 + hp.randint("selectkbest__k", 35))
                 #"selectkbest__score_func": hp.choice('selectkbest__score_func',
                 #                                    [f_regression, mutual_info_regression])
                 }
    elif model_type == "Ridge":
        model = linear_model.Ridge(alpha=0.1)
        space = {
            "model__alpha": hp.loguniform('model__alpha',
                                                     np.log(0.0001), np.log(10)),
            "selectkbest__k": (1 + hp.randint("selectkbest__k", 35))
            #"selectkbest__score_func": hp.choice('selectkbest__score_func',
            #                                     [f_regression, mutual_info_regression])
        }
    elif model_type == "Lasso":
        model = linear_model.Ridge(alpha=0.1)
        space = {
            "model__alpha": hp.loguniform('model__alpha',
                                                     np.log(0.0001), np.log(10)),
            "selectkbest__k": (1 + hp.randint("selectkbest__k", 35))
            #"selectkbest__score_func": hp.choice('selectkbest__score_func',
            #                                     [f_regression, mutual_info_regression])
        }

    else:
        model = RandomForestRegressor()
        space = {"model__n_estimators": (100 + hp.randint("model__n_estimators", 900)),
                 "model__max_depth": (4 + hp.randint("model__max_depth", 16)),
                 "selectkbest__k" : (1 + hp.randint("selectkbest__k", 35)),
                 #"selectkbest__score_func": hp.choice('selectkbest__score_func'
                 #                                     [f_regression, mutual_info_regression])
                }

    # Pipeline + Model
    full_model = FullModelClass(processing_pipeline, model)
    full_model.hyperopt(features=X, target=y, parameter_space=space, cv=5, max_evals=1000)

    ###### Evaluate Model
    X = df_train_eval.drop(columns='SalePrice')
    y = df_train_eval[['SalePrice']]

    y_pred = full_model.predict(X)


    # The mean squared error
    #evaluation_df = pd.concat([y, y_pred], axis=0)

    evaluation_df = y.copy()
    evaluation_df["SalePrice_pred"] = y_pred
    evaluation_df.to_csv("{}/data/evaluation_df.csv".format(dir_path), index=False)

    error = mean_squared_error(y, y_pred)
    print("Mean squared error: %.6f" % error)
    print("Root Mean squared error: %.6f" % np.sqrt(error))


    ###### PREDICTION KAGGLE
    # Final Train
    final_df_train = pd.concat([df_train, df_train_eval])
    X = final_df_train.drop(columns='SalePrice')
    y = final_df_train[['SalePrice']]
    full_model.fit_model_pipe(X, y)

    # Prediction
    X_test = pd.read_csv("{}/data/test.csv".format(dir_path))
    y_pred = full_model.predict(X_test)

    # Submission
    submission = X_test[['Id']]
    submission.insert(1, "SalePrice", y_pred, True)
    submission = transform_exp(submission, 'SalePrice')
    submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

    print(submission)
