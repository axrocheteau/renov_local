
import numpy as np
import pyspark as ps
import matplotlib as matplot
import sklearn as sk

# score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# copy
from copy import deepcopy
# linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

# random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# XGboost
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

Model = XGBRegressor | XGBClassifier | RandomForestClassifier | RandomForestRegressor | Ridge | LogisticRegression | HistGradientBoostingRegressor | HistGradientBoostingClassifier


def iterate_params(current: list[int], max_hyper: list[int]) -> list[int]:
    '''get next coverall of hyperparams'''
    for i, (max, curr) in enumerate(zip(max_hyper, current)):
        if max == curr:
            current[i] = 0
        else:
            current[i] += 1
            break
    return current


def choose_params(current: list[int], hyperparams: dict[str, list[int | str]]) -> dict[str, int | str]:
    '''create dictionary of parameters given the chosen ones'''
    hyper = {}
    for hyper_nb, (hyper_name, hyper_choices) in zip(current, hyperparams.items()):
        hyper[hyper_name] = hyper_choices[hyper_nb]
    return hyper


def nb_possibility(max_hyper: list[int]) -> int:
    '''nb of possibility for all hyperparams'''
    total = 1
    for nb_poss in max_hyper:
        total *= (nb_poss + 1)
    return total


def train_hyper(hyperparams: dict[str, list[int | str]], model: Model, X: np.ndarray, y: np.ndarray, split: ShuffleSplit, random_state: int) -> tuple[Model, float, dict[str, int | str], dict[tuple[int | str], float]]:
    '''training the model with given hyperparams'''
    search = HalvingRandomSearchCV(model(), param_distributions=hyperparams, n_candidates=100,
                                   random_state=random_state, cv=split, verbose=1, error_score=0)
    search.fit(X, y)
    scores = {}
    for params, score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score']):
        scores[tuple([param_value for param_value in params.values()])] = score
    best_model = search.best_estimator_
    best_score = search.best_score_
    best_params = search.best_params_
    return (best_model, best_score, best_params, scores)
