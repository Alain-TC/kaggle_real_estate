import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, fmin, Trials, space_eval
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit_transform(self, X, y=None, **fit_params):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.out_of_fold_predictions = out_of_fold_predictions
        return out_of_fold_predictions

    def fit_meta(self, X, y):
        self.meta_model_ = Pipeline([('model', self.meta_model_)])
        self.meta_model_.fit(X, y)

    # Hyperopt
    def hyperopt(self, features, target, parameter_space, cv=3, max_evals=5):
        # Parameter Search
        def _objective(space):
            self.meta_model_.set_params(**space)
            scores = cross_val_score(self.meta_model_, features, target, cv=cv,
                                     scoring='neg_mean_squared_error')
            print(np.sqrt(-np.mean(scores)))
            return np.sqrt(-np.mean(scores))

        tpe_trials = Trials()
        space = parameter_space
        best = fmin(fn=_objective, space=space, algo=tpe.suggest, max_evals=max_evals, verbose=True,
                    trials=tpe_trials)
        self.best_params = space_eval(space, best)
        print(self.best_params)

        # Training
        self.meta_model_.set_params(**self.best_params)
        self.meta_model_.fit(features, target)

        return self.best_params

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack(
            [np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in
             self.base_models_])
        return self.meta_model_.predict(meta_features)
