import logging
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import GridSearchCV



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
        self.pipe_feature_engineering.set_params(**self.best_params)
        self.pipe_feature_engineering.fit(features, target)
