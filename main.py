import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix
from preprocessing.transformers.log_target_transformer import transform_log, transform_exp
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest, chi2, f_regression, mutual_info_regression
from preprocessing.split_dataframe import split_dataframe_by_row
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from hyperopt import hp

from preprocessing.transformers.fillna_transformer import FillnaMeanTransformer
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
from modelisation.model import FullModelClass, create_model
from modelisation.config_hyperopt import get_config_hyperopt
import warnings
import pickle
from preprocessing.transformers.standardize_transformer import StandardizeTransformer
from preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder
from evaluation.metrics import evaluate_performance

warnings.filterwarnings('ignore')
#from preprocessing.transformers.normalize_transformer import NormalizeTransformer


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv("{}/data/train.csv".format(dir_path))


    # Transformation log(target)
    df_train = transform_log(df_train, 'SalePrice')

    # split Train/Eval
    df_train, df_train_eval = split_dataframe_by_row(df_train, 0.7)

    ## Preprocess data
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


    ## Pipeline
    # Preprocessing (outside crossval)
    preprocessing_pipeline = make_pipeline(SimpleOneHotEncoder(qualitative_columns),
                                           FillnaMeanTransformer(quantitative_columns),
                                           NormalizeTransformer(quantitative_columns))
    # Processing (inside crossval)
    processing_pipeline = make_pipeline(
        # KeepColumnsTransformer(quantitative_columns),

        # NormalizeTransformer(quantitative_columns)
        # StandardizeTransformer(quantitative_columns),

        # SelectKBest(score_func=mutual_info_regression, k=36),
        SelectKBest(score_func=f_regression, k=36)
    )

    ### Prepare Data
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]
    X = preprocessing_pipeline.fit_transform(X)

    ## Evaluate Model
    X_eval = df_train_eval.drop(columns='SalePrice')
    y_eval = df_train_eval[['SalePrice']]
    X_eval = preprocessing_pipeline.transform(X_eval)

    model_list = ["BayesianRidge", "GradientBoostingRegressor"]#, "ElasticNet"]#, "RandomForest"]#, "Ridge", "Lasso"]
    model_performances = []
    for model_name in model_list:
        # Split features and target



        model = create_model(model_name)

        space = get_config_hyperopt(model_name)

        # Pipeline + Model
        full_model = FullModelClass(processing_pipeline, model)
        full_model.hyperopt(features=X, target=y, parameter_space=space, cv=2, max_evals=3)

        ###### Evaluate Model
        y_pred = full_model.predict(X_eval)

        evaluation_df = y_eval.copy()
        evaluation_df["SalePrice_pred"] = y_pred
        evaluation_df.to_csv("{}/data/evaluation_df.csv".format(dir_path), index=False)

        # performances
        error = mean_squared_error(y_eval, y_pred)
        print("Mean squared error: %.6f" % error)
        print("Root Mean squared error: %.6f" % np.sqrt(error))
        performances = evaluate_performance(np.array(y_pred), np.array(y_eval))
        with open("models/performances/{}.json".format(model_name), 'w') as json_file:
            json_file.write(str(performances))

        model_performances.append((model_name, error))

    print("models performances: ")
    print(model_performances)
    # Best Model
    best_model_name = list(filter(lambda x: x[1]==min([y[1] for y in model_performances]), model_performances))[0][0]
    print(best_model_name)





    ###### PREDICTION KAGGLE
    # Final Train
    final_df_train = pd.concat([df_train, df_train_eval])
    X_final = final_df_train.drop(columns='SalePrice')
    y_final = final_df_train[['SalePrice']]
    X_final = preprocessing_pipeline.fit_transform(X_final)



    model = create_model(best_model_name)
    space = get_config_hyperopt(best_model_name)

    # Pipeline + Model
    full_model_final = FullModelClass(processing_pipeline, model)
    full_model_final.fit_model_pipe(X_final, y_final)

    # Store model
    full_model_filename = "{}/models/finalized_{}.sav".format(dir_path, best_model_name)
    pickle.dump(full_model_final, open(full_model_filename, 'wb'))

    # Store hyperparameters
    best_params = full_model_final.get_best_params()
    with open("models/hyperparameters/{}.json".format(best_model_name), 'w') as json_file:
        json_file.write(str(best_params))

