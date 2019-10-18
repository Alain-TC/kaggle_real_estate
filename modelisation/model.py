import logging
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import recall_score
from hyperopt import tpe, fmin
from hyperopt import Trials



class FullModelClass:
    def __init__(self, pipe_feature_engineering, model):
        self.model = model
        self.pipe_feature_engineering = pipe_feature_engineering
        # add a step with the model to the pipeline
        self.pipe_feature_engineering.steps.append(('model', self.model))
        self.best_params = None

    def fit_model_pipe(self, features, target):
        self.pipe_feature_engineering.fit(features, target)

    def predict(self, features):
        predictions = self.pipe_feature_engineering.predict(features)
        return predictions

    def fit_grid_search(self, features, target, parameters, cv=5):
        # Parameter Search
        self.search = GridSearchCV(self.pipe_feature_engineering, param_grid=parameters, cv=cv)
        self.search.fit(features, target)
        self.best_params = self.search.best_params_
        print("best parameters: {}".format(self.best_params))

        # Training
        self._set_params(self.best_params)
        self.pipe_feature_engineering.fit(features, target)

    def _set_params(self, parameters):
        print(parameters)
        self.pipe_feature_engineering.set_params(**parameters)



    def hyperopt(self, features, target, parameter_space, max_evals=5):
        def _objective(space):
            self._set_params(space)
            scores = cross_val_score(self.pipe_feature_engineering, features, target,cv=3)
            print(scores)
            print(np.mean(scores))
            return np.mean(scores)

        tpe_trials = Trials()
        space = parameter_space
        best = fmin(fn=_objective, space=space, algo=tpe.suggest, max_evals=max_evals, verbose=True,
                    trials=tpe_trials)
        return best, tpe_trials.best_trial['result']['loss']
