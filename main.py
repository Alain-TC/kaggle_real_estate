import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from preprocessing.transformers.column_selector_transformer import KeepColumnsTransformer
from preprocessing.transformers.dataframe_to_matrix_transformer import DataframeToMatrix

if __name__ == '__main__':
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/'))


    df_train = pd.read_csv("{}/data/train.csv".format(main_path))

    # Split features and target
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]

    # Preprocess data
    processing_pipeline = make_pipeline(KeepColumnsTransformer(['LotArea']), DataframeToMatrix())
    processing_pipeline.fit(X, y)
    X = processing_pipeline.transform(X)

    # Train Model
    regr = linear_model.LinearRegression()
    regr.fit(X,y)

    # Predict target
    y_pred = regr.predict(X)


    # Evaluate Model
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, y_pred))
