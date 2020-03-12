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
from preprocessing.transformers.fillna_transformer import FillnaMeanTransformer
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
from preprocessing.transformers.add_column_transformer import CreateTotalSFTransformer
from preprocessing.transformers.box_cox_transformer import BoxCoxTransformer
from preprocessing.split_dataframe import split_dataframe_by_row
from preprocessing.transformers.column_selector_transformer import ExcludeColumnsTransformer
from preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder
from preprocessing.outlier_detection import remove_outliers

from evaluation.metrics import evaluate_performance

from modelisation.config_columns import get_config_columns
from modelisation.model import FullModelClass, create_model
from modelisation.config_hyperopt import get_config_hyperopt

warnings.filterwarnings('ignore')

full_train = True

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv("{}/data/train.csv".format(dir_path))

    # Transformation log(target)
    df_train = transform_log(df_train, 'SalePrice')

    # split Train/Eval
    df_train, df_train_eval = split_dataframe_by_row(df_train, 0.7)

    # Preprocess data
    columns_config = get_config_columns()
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]
    all_qualitative_columns = qualitative_columns + semi_quali_columns

    df_train = remove_outliers(df_train, columns_config)
    df_train_eval = remove_outliers(df_train_eval, columns_config)

    # PIPELINE
    # Preprocessing (outside crossval)
    preprocessing_pipeline = make_pipeline(ExcludeColumnsTransformer(["Id"]),
                                           CreateTotalSFTransformer(),
                                           BoxCoxTransformer(quantitative_columns))

    # Processing (inside crossval)
    processing_pipeline = make_pipeline(
        FillnaMeanTransformer(quantitative_columns),
        NormalizeTransformer(quantitative_columns),
        #LeaveOneOutEncoder(semi_quali_columns),
        TargetEncoder(semi_quali_columns),
        SimpleOneHotEncoder(qualitative_columns),
        # ShowTransformer("end"),
        # KeepColumnsTransformer(quantitative_columns),
        # NormalizeTransformer(quantitative_columns)
        # StandardizeTransformer(quantitative_columns),
        SelectKBest(score_func=mutual_info_regression, k=36)
        #SelectKBest(score_func=f_regression, k=106)
    )

    # Prepare Data Training
    X = df_train.drop(columns='SalePrice')
    y = df_train[['SalePrice']]
    X = preprocessing_pipeline.fit_transform(X, y)     # Preprocessing before beacause out of the crossval

    # Prepare Data Evaluation
    X_eval = df_train_eval.drop(columns='SalePrice')
    y_eval = df_train_eval[['SalePrice']]
    X_eval = preprocessing_pipeline.transform(X_eval)   # Preprocessing before beacause out of the crossval

    # Prepare Data#Final Train
    final_df_train = pd.concat([df_train, df_train_eval])
    X_final = final_df_train.drop(columns='SalePrice')
    y_final = final_df_train[['SalePrice']]

    model_list = ["GradientBoostingRegressor"]
    #, "RandomForest", "BayesianRidge"]#, "Lasso"]#, "Ridge", "GradientBoostingRegressor", "ElasticNet"]
    model_performances = []

    for model_name in model_list:
        model = create_model(model_name)
        space = get_config_hyperopt(model_name)

        # Pipeline + Model
        full_model = FullModelClass(processing_pipeline, model)
        full_model.hyperopt(features=X, target=y, parameter_space=space, cv=3, max_evals=20)

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

        if full_train:
            # FULL TRAIN
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

            # Store model
            full_model_filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
            pickle.dump(full_model_final, open(full_model_filename, 'wb'))

    print("models performances: ")
    print(model_performances)
    # Best Model
    best_model_name = list(filter(lambda x: x[1] == min([y[1] for y in model_performances]), model_performances))[0][0]
    print(best_model_name)
