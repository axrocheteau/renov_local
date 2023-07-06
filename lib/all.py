# librairies
import numpy as np
import pyspark as ps
import matplotlib.pyplot as plt


# prepare data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit

# score
from sklearn.model_selection import cross_val_predict

# copy
from copy import deepcopy

# import other functions
from prepare_data import *
from train import *
from usefull import *
from show import *

# linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

# random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# XGboost
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier

# quick XGboost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

Dataframe = ps.sql.dataframe.DataFrame


Model = XGBRegressor | XGBClassifier | RandomForestClassifier | RandomForestRegressor | Ridge | LogisticRegression | HistGradientBoostingRegressor | HistGradientBoostingClassifier


def all_in_one(df: Dataframe,
               dictionary: Dataframe,
               categorical_feature: list[int],
               col_X_hots: list[list[str]],
               col_X_not_hots: list[list[str]],
               col_y: list[str],
               y_hot: bool,
               scale: bool,
               hyperparams_models: list[dict[str, int | str]],
               models: Model,
               degree_poly: int = 1,
               random_state: int = 42,
               test_size: float = 0.4,
               show: bool = True) -> list[Model]:
    '''handle prepare data, training, and show the results'''
    # store best_models
    best_models = {}

    # prepare to plot
    if show:
        f1, ax_result = plt.subplots(
            1, len(models), figsize=(20, 5), sharey=True)
        f2, ax_hyper = plt.subplots(
            1, len(models), figsize=(20, 5), sharey=True)
        f3, ax_importance = plt.subplots(1, len(models), figsize=(20, 20))

    # iterate over models
    for i, (col_X_hot, col_X_not_hot, hyperparams, (model_name, model)) in enumerate(zip(col_X_hots, col_X_not_hots, hyperparams_models, models.items())):
        # prepare data
        X, y, labels = prepare_dataset(
            df, dictionary, col_X_hot, col_X_not_hot, col_y, y_hot, scale)
        poly = PolynomialFeatures(degree_poly)
        X_transformed = poly.fit_transform(X)
        cv = ShuffleSplit(n_splits=4, test_size=test_size,
                          random_state=random_state)

        # training models
        best_model, best_score, best_params, scores = train_hyper(
            hyperparams, model, X_transformed, y, cv, random_state, categorical_feature)
        best_models[model_name] = [deepcopy(best_model), best_score, best_params]

        # plot results
        print(best_score, best_params)
        y_pred = cross_val_predict(best_model, X_transformed, y, cv=4)
        if show:
            if len(models) > 1:
                if len(np.unique(y)) > 10:
                    show_result(y_pred, y, ax_result[i], model_name)
                else:
                    show_matrix(y_pred, y, ax_result[i], model_name)
                show_hyperparam_opti(scores, hyperparams,
                                     ax_hyper[i], model_name)
                show_importance(best_model, labels,
                                ax_importance[i], model_name)
            else:
                if len(np.unique(y)) > 10:
                    show_result(y_pred, y, ax_result, model_name)
                else:
                    show_matrix(y_pred, y, ax_result, model_name)
                show_hyperparam_opti(scores, hyperparams,
                                     ax_hyper, model_name)
                show_importance(best_model, labels, ax_importance, model_name)

    return best_models
