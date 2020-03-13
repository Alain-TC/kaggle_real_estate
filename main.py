import os
import warnings
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_regression
from category_encoders import TargetEncoder

from preprocessing.transformers.log_target_transformer import transform_log
from preprocessing.transformers.fillna_transformer import FillnaMeanTransformer, FillnaMeanMatrixTransformer
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
from preprocessing.transformers.add_column_transformer import CreateTotalSFTransformer
from preprocessing.transformers.box_cox_transformer import BoxCoxTransformer
from preprocessing.split_dataframe import split_dataframe_by_row
from preprocessing.transformers.column_selector_transformer import ExcludeColumnsTransformer
from preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder
from preprocessing.outlier_detection import remove_outliers
from preprocessing.transformers.show_transformer import ShowTransformer
from preprocessing.transformers.log_target_transformer import transform_exp

from evaluation.metrics import evaluate_performance

from modelisation.config_columns import get_config_columns
from modelisation.model import FullModelClass, create_model
from modelisation.config_hyperopt import get_config_hyperopt

warnings.filterwarnings('ignore')

HYPEROPT = True
FULLTRAIN = True
PREDICT = True
model_list = ["LightGBM"]
#model_list = ["GradientBoostingRegressor", "ElasticNet"]

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_total = pd.read_csv("{}/data/train.csv".format(dir_path))

    # Preprocess data
    columns_config = get_config_columns()
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]
    all_qualitative_columns = qualitative_columns + semi_quali_columns

    # PIPELINE
    # Preprocessing (outside crossval)
    preprocessing_pipeline = make_pipeline(ExcludeColumnsTransformer(["Id"]),
                                           CreateTotalSFTransformer(),
                                           BoxCoxTransformer(quantitative_columns))

    # Remove outliers
    df_total = remove_outliers(df_total, columns_config)
    # Transformation log(target)
    df_total = transform_log(df_total, 'SalePrice')

    # Processing (inside crossval)
    processing_pipeline = make_pipeline(
        FillnaMeanTransformer(quantitative_columns),
        NormalizeTransformer(quantitative_columns),
        # LeaveOneOutEncoder(semi_quali_columns),
        TargetEncoder(semi_quali_columns),
        SimpleOneHotEncoder(qualitative_columns),
        FillnaMeanMatrixTransformer(),
        # ShowTransformer("middle"),
        # ShowTransformer("end"),
        # KeepColumnsTransformer(quantitative_columns),
        # NormalizeTransformer(quantitative_columns)
        # StandardizeTransformer(quantitative_columns),
        # SelectKBest(score_func=mutual_info_regression, k=36),
        # SelectKBest(score_func=f_regression, k=106),
        # ShowTransformer("end")
    )

    if HYPEROPT:
        # split Train/Eval
        df_train, df_train_eval = split_dataframe_by_row(df_total, 0.9)
        # Remove outliers
        # df_train = remove_outliers(df_train, columns_config)
        # df_train_eval = remove_outliers(df_train_eval, columns_config)

        # Prepare Data Training
        X = df_train.drop(columns='SalePrice')
        y = df_train[['SalePrice']]
        X = preprocessing_pipeline.fit_transform(X, y)  # Preprocessing before beacause out of the crossval

        # Prepare Data Evaluation
        X_eval = df_train_eval.drop(columns='SalePrice')
        y_eval = df_train_eval[['SalePrice']]
        X_eval = preprocessing_pipeline.transform(X_eval)  # Preprocessing before beacause out of the crossval

        # , "RandomForest", "BayesianRidge"]#, "Lasso"]#, "Ridge", "GradientBoostingRegressor", "ElasticNet"]
        model_performances = []

        for model_name in model_list:
            model = create_model(model_name)
            space = get_config_hyperopt(model_name)

            # Pipeline + Model
            full_model = FullModelClass(processing_pipeline, model)
            full_model.hyperopt(features=X, target=y, parameter_space=space, cv=3, max_evals=500)

            # Store hyperparameters
            best_params = full_model.get_best_params()
            with open("models/hyperparameters/{}.json".format(model_name), 'w') as json_file:
                json_file.write(json.dumps(best_params))

            # Evaluate Model
            y_pred = full_model.predict(X_eval)
            evaluation_df = y_eval.copy()
            evaluation_df["SalePrice_pred"] = y_pred
            evaluation_df.to_csv("{}/data/evaluation_df.csv".format(dir_path), index=False)

            # performances
            error = mean_squared_error(y_eval, y_pred)
            print("Root Mean squared error: %.6f" % np.sqrt(error))
            performances = evaluate_performance(np.array(y_pred), np.array(y_eval))
            with open("models/performances/{}.json".format(model_name), 'w') as json_file:
                json_file.write(str(performances))
            model_performances.append((model_name, np.sqrt(error)))

        print("models performances: ")
        print(model_performances)
        # Best Model
        best_model_name = \
            list(filter(lambda x: x[1] == min([y[1] for y in model_performances]), model_performances))[0][0]
        print(best_model_name)

    if FULLTRAIN:
        for model_name in model_list:
            # Prepare Data#Final Train

            X_final = df_total.drop(columns='SalePrice')
            y_final = df_total[['SalePrice']]

            model = create_model(model_name)

            # Pipeline + Model
            full_model_final = FullModelClass(processing_pipeline, model)
            # Add preprocessing to model
            full_model_final._enrich_pipe_upstream(preprocessing_pipeline)

            # Set parameters
            with open("models/hyperparameters/{}.json".format(model_name)) as json_file:
                best_params = json.load(json_file)
            print(best_params)

            full_model_final._set_params(best_params)
            full_model_final.fit_model_pipe(X_final, y_final)

            # performances
            y_pred = full_model_final.predict(X_final)
            error = mean_squared_error(y_final, y_pred)
            print("Root Mean squared error: %.6f" % np.sqrt(error))

            # Store model
            full_model_filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
            pickle.dump(full_model_final, open(full_model_filename, 'wb'))

    if PREDICT:
        X_test = pd.read_csv("{}/data/test.csv".format(dir_path))
        y_pred_list = []
        for model_name in model_list:
            filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
            # load the model from disk
            loaded_model = pickle.load(open(filename, 'rb'))
            y_pred = loaded_model.predict(X_test)
            y_pred_list.append(y_pred)

        y_pred_list = pd.DataFrame(y_pred_list)
        y_pred_list = y_pred_list.transpose()
        final_y_pred = y_pred_list.mean(axis=1)

        # Submission
        submission = X_test[['Id']]
        submission.insert(1, "SalePrice", final_y_pred, True)
        submission = transform_exp(submission, 'SalePrice')
        submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

        print(submission)
