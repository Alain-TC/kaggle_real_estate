import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .modelisation.config_columns import get_config_columns
from .modelisation.config_hyperopt import get_config_hyperopt
from .modelisation.pipelines import pipe_preprocessing, pipe_processing
from .modelisation.stack_models import StackingAveragedModels
from .preprocessing.outlier_detection import remove_outliers
from .preprocessing.split_dataframe import split_dataframe_by_row
from .preprocessing.transformers.log_target_transformer import transform_exp
from .preprocessing.transformers.log_target_transformer import transform_log

from .evaluation.metrics import evaluate_performance
from .modelisation.model import FullModelClass, create_model

from .connection.io import write_json, read_json

warnings.filterwarnings('ignore')

HYPEROPT = True
FULLTRAIN = True
PREDICT = False
STACKING = True
STACKING_HYPEROPT = False

SAVE_MODELS = True

model_list = ["Ridge", "ElasticNet"]
# model_list = ["GradientBoostingRegressor", "ElasticNet", "LightGBM", "BayesianRidge", "Lasso", "Ridge", "RandomForest",
#              "XGBRegressor"]  # , "SVR"]

meta_model_name = "ElasticNet"

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_total = pd.read_csv("{}/data/train.csv".format(dir_path))

    # Preprocess data
    columns_config = get_config_columns()

    # PIPELINE
    # Preprocessing (outside crossval)
    preprocessing_pipeline = pipe_preprocessing(columns_config)
    # Processing (inside crossval)
    processing_pipeline = pipe_processing(columns_config)

    # Remove outliers
    df_total = remove_outliers(df_total, columns_config)
    # Transformation log(target)
    df_total = transform_log(df_total, 'SalePrice')
    df_full_train = df_total.copy()
    df_stacking_train, df_stacking_test = split_dataframe_by_row(df_total, 0.7, seed=42)

    if HYPEROPT:
        # split Train/Eval
        df_train, df_train_eval = split_dataframe_by_row(df_total, 0.9)

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
            full_model.hyperopt(features=X, target=y, parameter_space=space, cv=3, max_evals=100)

            # Store hyperparameters
            best_params = full_model.get_best_params()
            print(best_params)
            if SAVE_MODELS:
                write_json(item=best_params, path="{}/models/hyperparameters/{}.json".format(dir_path, model_name))

            # Evaluate Model
            y_pred = full_model.predict(X_eval)
            evaluation_df = y_eval.copy()
            evaluation_df["SalePrice_pred"] = y_pred
            evaluation_df.to_csv("{}/data/evaluation_df.csv".format(dir_path), index=False)

            # performances
            error = mean_squared_error(y_eval, y_pred)
            print("Root Mean squared error: %.6f" % np.sqrt(error))
            performances = evaluate_performance(np.array(y_pred), np.array(y_eval))
            write_json(item=performances, path="{}/models/performances/{}.json".format(dir_path, model_name))
            model_performances.append((model_name, np.sqrt(error)))

        print("models performances: ")
        print(model_performances)
        # Best Model
        best_model_name = \
            list(filter(lambda x: x[1] == min([y[1] for y in model_performances]), model_performances))[0][0]
        print(best_model_name)

    if FULLTRAIN:
        # Prepare Data#Final Train
        X_final = df_full_train.drop(columns='SalePrice')
        y_final = df_full_train[['SalePrice']]

        for model_name in model_list:
            model = create_model(model_name)

            # Pipeline + Model
            full_model_final = FullModelClass(processing_pipeline, model)
            # Add preprocessing to model
            full_model_final._enrich_pipe_upstream(preprocessing_pipeline)

            # Set parameters
            best_params = read_json("{}/models/hyperparameters/{}.json".format(dir_path, model_name))
            print(best_params)

            full_model_final._set_params(best_params)
            full_model_final.fit_model_pipe(X_final, y_final)

            # performances
            y_pred = full_model_final.predict(X_final)
            error = mean_squared_error(y_final, y_pred)
            print("Root Mean squared error: %.6f" % np.sqrt(error))

            # Store model
            if SAVE_MODELS:
                full_model_filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
                pickle.dump(full_model_final, open(full_model_filename, 'wb'))

    if PREDICT:
        X_test = pd.read_csv("{}/data/test.csv".format(dir_path))
        y_pred_list = []
        for model_name in model_list:
            filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
            print(filename)
            # load the model from disk
            loaded_model = pickle.load(open(filename, 'rb'))
            y_pred = loaded_model.predict(X_test)
            y_pred = [float(x) for x in y_pred]
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

    if STACKING:
        # Prepare Data Training
        X = df_full_train.drop(columns='SalePrice')
        y = df_full_train['SalePrice']

        X_test = pd.read_csv("{}/data/test.csv".format(dir_path))

        base_models = []
        for model_name in model_list:
            model = create_model(model_name)
            model = FullModelClass(processing_pipeline, model)
            model._enrich_pipe_upstream(preprocessing_pipeline)
            best_params = read_json("{}/models/hyperparameters/{}.json".format(dir_path, model_name))

            model._set_params(best_params)
            base_models.append(model.return_pipeline())

        # Meta Learner
        meta_model = create_model(meta_model_name)

        # Stacked Model
        stacked_model = StackingAveragedModels(base_models=base_models, meta_model=meta_model)
        space = get_config_hyperopt(meta_model_name)

        X = stacked_model.fit_transform(X, y)

        stacked_model_filename = "{}/models/finalized_meta_{}.sav".format(dir_path, meta_model_name)
        if STACKING_HYPEROPT:
            stacked_model.hyperopt(features=X, target=y, parameter_space=space, cv=3, max_evals=100)
            best_params = stacked_model.get_best_params()
            if SAVE_MODELS:
                write_json(item=best_params,
                       path="{}/models/hyperparameters/stacked_{}.json".format(dir_path, meta_model_name))
                # Store model
                pickle.dump(stacked_model, open(stacked_model_filename, 'wb'))
        else:
            # Set parameters
            best_params = read_json("{}/models/hyperparameters/stacked_{}.json".format(dir_path, meta_model_name))
            stacked_model._set_params(best_params)
            stacked_model.fit_meta(X, y)

        y_pred = stacked_model.predict(X_test)

        # Submission
        submission = X_test[['Id']]
        submission.insert(1, "SalePrice", y_pred, True)
        submission = transform_exp(submission, 'SalePrice')
        submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

        print(submission)
