import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix
from preprocessing.transformers.log_target_transformer import transform_log, transform_exp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_train = pd.read_csv("{}/data/train.csv".format(dir_path))
    df_train.fillna(0, inplace=True)
    df_train = transform_log(df_train, 'SalePrice')

    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]

    # Preprocess data
    processing_pipeline = make_pipeline(KeepColumnsTransformer(["MSSubClass","LotFrontage","OverallQual","OverallCond",
                                                                "YearBuilt","YearRemodAdd","MasVnrArea",
                                                                "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                                                                "1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea",
                                                                "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
                                                                "BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd",
                                                                "Fireplaces","GarageYrBlt","GarageCars","GarageArea",
                                                                "WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch",
                                                                "ScreenPorch","PoolArea","MiscVal","MoSold","YrSold",
                                                                "LotArea"]), DataframeToMatrix())
    processing_pipeline.fit(X, y)
    X = processing_pipeline.transform(X)

    # Train Model
    parameters = {'max_depth': [2, 5, 8], 'n_estimators': [50, 100, 500]}
    regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=1000)
    clf = GridSearchCV(regr, parameters, cv=5)
    clf.fit(X, y)

    # Predict target
    y_pred = clf.predict(X)


    # Evaluate Model


    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, y_pred))


    # PREDICTION
    df_test = pd.read_csv("{}/data/test.csv".format(dir_path))
    df_test.fillna(0, inplace=True)
    X = df_test

    X = processing_pipeline.transform(X)
    y_pred = clf.predict(X)

    submission = df_test[['Id']]
    submission.insert(1, "SalePrice", y_pred, True)
    submission = transform_exp(submission, 'SalePrice')
    submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

    print(submission)

