import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix
from preprocessing.transformers.log_target_transformer import transform_log, transform_exp
from preprocessing.split_dataframe import split_dataframe_by_row
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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

    processing_pipeline = make_pipeline(SimpleOneHotEncoder(qualitative_columns),
                                        KeepColumnsTransformer(quantitative_columns),
                                        FillnaMeanTransformer(quantitative_columns),
                                        #NormalizeTransformer(quantitative_columns)
                                        StandardizeTransformer(quantitative_columns),
                                        DataframeToMatrix())

    # Model
    model = RandomForestRegressor()

    # Pipeline + Model
    full_model = FullModelClass(processing_pipeline, model)

    ###### Test de l'hyperopt
    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]


    space = {"model__n_estimators": (1+hp.randint("n_estimators_hp", 199)),
            "model__max_depth": (1+hp.randint("max_depth_hp", 19))
            }
    full_model.hyperopt(X, y, space, 5)



    ###### Entrainement et grid_search
    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]

    parameters = {'model__max_depth': [2 * (1 + x) for x in range(5)], 'model__n_estimators': [100, 500, 1000, 1500]}
    parameters = {'model__max_depth': [8], 'model__n_estimators': [100]}
    full_model.fit_grid_search(features=X, target=y, parameters=parameters)

    ###### Evaluate Model
    X = df_train_eval.drop(columns='SalePrice')
    y = df_train_eval[['SalePrice']]

    y_pred = full_model.predict(X)


    # The mean squared error
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
