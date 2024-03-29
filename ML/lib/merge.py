# librairies
import numpy as np
import pyspark as ps
import matplotlib.pyplot as plt

# prepare data
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import SMOTE

# score
from sklearn.model_selection import cross_val_predict

# copy
from copy import deepcopy

# import other functions
from lib.prepare_data import prepare_dataset, get_labels
from lib.train import train_hyper
from lib.show import show_hyperparam_opti, show_importance, show_matrix, show_result

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

# typings
Dataframe = ps.sql.dataframe.DataFrame
Model = (
    XGBRegressor
    | XGBClassifier
    | RandomForestClassifier
    | RandomForestRegressor
    | Ridge
    | LogisticRegression
    | HistGradientBoostingRegressor
    | HistGradientBoostingClassifier
)


def prepare_train_show(
    df: Dataframe,
    dictionary: Dataframe,
    categorical_feature: list[int],
    col_X_hots: list[list[str]],
    col_X_not_hots: list[list[str]],
    col_y: list[str],
    y_hot: bool,
    scale: bool,
    hyperparams_models: list[dict[str, int | str]],
    models: Model,
    random_state: int = 42,
    test_size: float = 0.4,
    show: bool = True,
    verbose=0,
    scoring: str = None,
    smote: bool = False,
    treshold: int = 20000,
) -> list[Model]:
    """handle prepare data, training, and show the results"""
    # store best_models
    best_models = {}

    # prepare to plot
    if show:
        f1, ax_result = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
        f2, ax_hyper = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
        f3, ax_importance = plt.subplots(1, len(models), figsize=(20, 20))

    # iterate over models
    for i, (col_X_hot, col_X_not_hot, hyperparams, (model_name, model)) in enumerate(
        zip(col_X_hots, col_X_not_hots, hyperparams_models, models.items())
    ):
        # prepare data
        X_original, y_original, labels = prepare_dataset(
            df, dictionary, col_X_hot, col_X_not_hot, col_y, y_hot, scale, treshold
        )
        if smote:
            sm = SMOTE(random_state=random_state)
            X, y = sm.fit_resample(X_original, y_original)

        cv = ShuffleSplit(n_splits=4, test_size=test_size, random_state=random_state)
        labels_matrix = get_labels(dictionary, col_y[0])["meaning"]

        # training models
        if smote:
            best_model, best_score, best_params, scores = train_hyper(
                hyperparams,
                model,
                X,
                y,
                cv,
                random_state,
                categorical_feature,
                verbose,
                scoring,
            )
        else:
            best_model, best_score, best_params, scores = train_hyper(
                hyperparams,
                model,
                X_original,
                y_original,
                cv,
                random_state,
                categorical_feature,
                verbose,
                scoring,
            )
        best_models[model_name] = [deepcopy(best_model), best_score, best_params]

        # plot results
        print(best_score, best_params)
        y_pred = cross_val_predict(best_model, X_original, y_original, cv=4)
        if show:
            if len(models) > 1:
                if len(np.unique(y_original)) > 10:
                    show_result(y_pred, y_original, ax_result[i], model_name)
                else:
                    show_matrix(
                        y_pred, y_original, ax_result[i], model_name, labels_matrix
                    )
                show_hyperparam_opti(scores, hyperparams, ax_hyper[i], model_name)
                show_importance(best_model, labels, ax_importance[i], model_name)
            else:
                if len(np.unique(y_original)) > 10:
                    show_result(y_pred, y_original, ax_result, model_name)
                else:
                    show_matrix(
                        y_pred, y_original, ax_result, model_name, labels_matrix
                    )
                show_hyperparam_opti(scores, hyperparams, ax_hyper, model_name)
                show_importance(best_model, labels, ax_importance, model_name)

    return best_models
