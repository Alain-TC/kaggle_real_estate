from hyperopt import hp
import numpy as np


def get_config_hyperopt(model_name):
    transformer_space = {"selectkbest__k": (10 + hp.randint("selectkbest__k", 70))
                         # "selectkbest__score_func": hp.choice('selectkbest__score_func' [f_regression, mutual_info_regression])
                         }

    model_spaces = {
        "Linear": {},
        "Lasso": {"model__alpha": hp.loguniform('model__alpha', np.log(0.0001), np.log(0.01))},
        "Ridge": {"model__alpha": hp.loguniform('model__alpha', np.log(0.0001), np.log(0.01))},
        "ElasticNet": {"model__alpha": hp.loguniform('model__alpha', np.log(0.00001), np.log(0.01)),
                       "model__l1_ratio": hp.uniform('model__l1_ratio', 0, 1)},
        "RandomForest": {"model__n_estimators": (100 + hp.randint("model__n_estimators", 900)),
                         "model__max_depth": (4 + hp.randint("model__max_depth", 16))}
    }
    transformer_space.update(model_spaces[model_name])
    return transformer_space
