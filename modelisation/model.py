import logging
from sklearn.model_selection import KFold
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import recall_score
from hyperopt import tpe, fmin, Trials, space_eval
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression



def create_model(model_name):
    if model_name == 'Linear':
        return linear_model.LinearRegression()
    elif model_name == 'Lasso':
        return linear_model.Lasso()
    elif model_name == 'Ridge':
        return linear_model.Ridge()
    elif model_name == 'ElasticNet':
        return linear_model.ElasticNet()
    elif model_name == 'RandomForest':
        return RandomForestRegressor()
    elif model_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor()
    elif model_name == 'BayesianRidge':
        return BayesianRidge()
    else:
        raise KeyError("Incorrect model_name, received '%s' instead." % model_name)


class FullModelClass:
    def __init__(self, pipe_feature_engineering, model, preprocess=None):
        self.model = model
        self.pipe_feature_engineering = sklearn.base.clone(pipe_feature_engineering)
        # add a step with the model to the pipeline
        self.pipe_feature_engineering.steps.append(('model', self.model))

        self.params_list = []
        self.best_params = None

    def _enrich_pipe_upstream(self, upstream_pipe):
        self.pipe_feature_engineering.steps.insert(0, ['preprocessing', upstream_pipe])


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
        print("parameters")
        print(parameters)
        self.pipe_feature_engineering.set_params(**parameters)

    def get_best_params(self):
        return self.best_params


    def hyperopt(self, features, target, parameter_space, cv=3, max_evals=5):
        # Parameter Search
        def _objective(space):
            self._set_params(space)
            scores = cross_val_score(self.pipe_feature_engineering, features, target,cv=cv, scoring='neg_mean_squared_error')
            print(np.sqrt(-np.mean(scores)))
            return np.sqrt(-np.mean(scores))

        tpe_trials = Trials()
        space = parameter_space
        best = fmin(fn=_objective, space=space, algo=tpe.suggest, max_evals=max_evals, verbose=True,
                    trials=tpe_trials)
        self.best_params = space_eval(space, best)
        print(self.best_params)

        # Training
        self._set_params(self.best_params)
        self.pipe_feature_engineering.fit(features, target)

        return self.best_params

        #return best, tpe_trials.best_trial['result']['loss']


