import os
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix
from preprocessing.transformers.log_target_transformer import transform_log, transform_exp
from preprocessing.split_dataframe import split_dataframe_by_row

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preprocessing.transformers.fillna_transformer import FillnaTransformer



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_train = pd.read_csv("{}/data/train.csv".format(dir_path))
    df_train.fillna(0, inplace=True)

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
    processing_pipeline = make_pipeline(KeepColumnsTransformer(quantitative_columns),
                                        FillnaTransformer(quantitative_columns), DataframeToMatrix())

    ###### Entrainement et grid_search
    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]

    processing_pipeline.fit(X, y)
    X = processing_pipeline.transform(X)

    # Train Model
    parameters = {'max_depth': [2*(1+x) for x in range(5)], 'n_estimators': [100, 500, 1000, 1500]}
    regr = RandomForestRegressor()
    clf = GridSearchCV(regr, parameters, cv=3)
    clf.fit(X, y)

    best_params = clf.get_params()
    print("best_params: {}".format(best_params))

    ###### Evaluate Model
    X = df_train_eval.drop(columns='SalePrice')
    y = df_train_eval[['SalePrice']]

    processing_pipeline.fit(X, y)
    X = processing_pipeline.transform(X)

    y_pred = clf.predict(X)

    error = mean_squared_error(y, y_pred)


    # The mean squared error
    print("Mean squared error: %.6f" % error)
    print("Root Mean squared error: %.6f" % np.sqrt(error))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.6f' % r2_score(y, y_pred))


    ###### PREDICTION
    max_depth = best_params['estimator__max_depth']
    n_estimators = best_params['estimator__n_estimators']
    regr = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)

    final_df_train = pd.concat([df_train, df_train_eval])

    X = final_df_train.drop(columns='SalePrice')
    y = final_df_train[['SalePrice']]

    processing_pipeline.fit(X, y)
    X = processing_pipeline.transform(X)

    regr.fit(X, y)

    df_test = pd.read_csv("{}/data/test.csv".format(dir_path))
    df_test.fillna(0, inplace=True)
    X_test = df_test

    X_test = processing_pipeline.transform(X_test)
    y_pred = regr.predict(X_test)

    submission = df_test[['Id']]
    submission.insert(1, "SalePrice", y_pred, True)
    submission = transform_exp(submission, 'SalePrice')
    submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

    print(submission)

