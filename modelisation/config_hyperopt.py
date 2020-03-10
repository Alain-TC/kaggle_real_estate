from hyperopt import hp
import numpy as np


def get_config_hyperopt(model_name):
    transformer_space = {
        "selectkbest__k": (1 + hp.randint("selectkbest__k", 78))
        # "selectkbest__score_func": hp.choice('selectkbest__score_func', [f_regression, mutual_info_regression])
        }

    model_spaces = {
        "Linear": {

        },
        "Lasso": {
            "model__alpha": hp.loguniform('model__alpha', np.log(0.0001), np.log(0.01))
        },
        "Ridge": {
            "model__alpha": hp.loguniform('model__alpha', np.log(0.0001), np.log(0.01))
        },
        "ElasticNet": {
            "model__alpha": hp.loguniform('model__alpha', np.log(0.00001), np.log(0.01)),
            "model__l1_ratio": hp.uniform('model__l1_ratio', 0, 1)
        },
        "RandomForest": {
            "model__n_estimators": (100 + hp.randint("model__n_estimators", 900)),
            "model__max_depth": (4 + hp.randint("model__max_depth", 16))},
        "GradientBoostingRegressor": {
            "model__n_estimators": (100 + hp.randint("model__n_estimators", 900)),
            #"model__n_estimators": (500),
            "model__learning_rate": hp.loguniform('model__learning_rate', np.log(0.001), np.log(1)),
            "model__subsample": hp.uniform('model__subsample', 0, 1)
        },
        "BayesianRidge" : {
            "model__alpha_1": hp.loguniform('model__alpha_1', np.log(0.0000001), np.log(0.01)),
            "model__alpha_2": hp.loguniform('model__alpha_2', np.log(0.0000001), np.log(0.01)),
            "model__lambda_1": hp.loguniform('model__lambda_1', np.log(0.0000001), np.log(0.01)),
            "model__lambda_2": hp.loguniform('model__lambda_2', np.log(0.0000001), np.log(0.01)),
            "model__n_iter": (100 + hp.randint("model__n_iter", 500)),
            "model__normalize": hp.choice('model__normalize', [True, False])
        }
    }
    transformer_space.update(model_spaces[model_name])
    return transformer_space
