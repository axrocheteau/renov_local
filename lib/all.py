# librairies
import numpy as np
import sklearn as sk
import pyspark as ps
import matplotlib.pyplot as plt
import matplotlib as matplot

# prepare data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit

# score
from sklearn.model_selection import cross_val_predict

# copy
from copy import deepcopy

#import other functions
from lib.prepare_data import *
from lib.train import *
from lib.usefull import *
from lib.show import *

#linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# XGboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

Dataframe = ps.sql.dataframe.DataFrame
Model = GradientBoostingRegressor | GradientBoostingClassifier | RandomForestClassifier | RandomForestRegressor | Ridge | LogisticRegression

# handle prepare data, training, and show the results
def all_in_one(df: Dataframe,
                dictionary: Dataframe,
                col_X_hots: list[list[str]],
                col_X_not_hots: list[list[str]],
                col_y: list[str],
                y_hot: bool,
                scale:bool,
                hyperparams_models: list[dict[str, int|str]],
                models: Model,
                degree_poly: int = 1,
                random_state: int = 42,
                test_size: float = 0.4,
                show: bool = True) -> list[Model]:

    # store best_models
    best_models = {}

    # prepare to plot
    if show:
        f1, ax_result = plt.subplots(1, len(models), figsize=(15,5), sharey= True)
        f2, ax_hyper = plt.subplots(1, len(models), figsize=(15,5), sharey= True)
        f3, ax_importance = plt.subplots(1, len(models), figsize=(15,20))

    #iterate over models
    for i, (col_X_hot, col_X_not_hot, hyperparams, (model_name, model)) in enumerate(zip(col_X_hots, col_X_not_hots, hyperparams_models, models.items())):
        # prepare data
        X, y, labels = prepare_dataset(df, dictionary, col_X_hot, col_X_not_hot, col_y, y_hot, scale)
        poly = PolynomialFeatures(degree_poly)
        X_transformed = poly.fit_transform(X)
        cv = ShuffleSplit(n_splits=4, test_size=test_size, random_state=random_state)
        
        # training models
        best_model, best_score, best_params, scores = train_hyper(hyperparams, model, X_transformed, y, cv)
        best_models[model_name] = [deepcopy(best_model), best_score]

        # plot results
        print(best_score, best_params)
        y_pred = cross_val_predict(best_model, X_transformed, y, cv=4)
        if show:
            if len(hyperparams) > 1:
                if len(np.unique(y)) > 10:
                    show_result(y_pred, y, ax_result[i], model_name)
                else:
                    show_matrix(y_pred, y, ax_result[i], model_name)
                show_hyperparam_opti(scores, hyperparams, ax_hyper[i], model_name)
                show_importance(best_model.fit(X_transformed[list(cv.split(X_transformed))[0][0]], y[list(cv.split(y))[0][0]]), labels, ax_importance[i], model_name)
            else:
                if len(np.unique(y)) > 10:
                    show_result(y_pred, y, ax_result, model_name)
                else:
                    show_matrix(y_pred, y, ax_result, model_name)
                show_hyperparam_opti(scores, hyperparams, ax_hyper[i], model_name)
                show_importance(best_model.fit(X_transformed[list(cv.split(X_transformed))[0][0]], y[list(cv.split(y))[0][0]]), labels, ax_importance, model_name)

    return best_models
