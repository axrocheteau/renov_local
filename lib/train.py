
import numpy as np
import pyspark as ps
import matplotlib as matplot
import sklearn as sk

# score
from sklearn.model_selection import cross_val_score

# copy
from copy import deepcopy
#linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# XGboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

Model = GradientBoostingRegressor | GradientBoostingClassifier | RandomForestClassifier | RandomForestRegressor | Ridge | LogisticRegression

# get next coverall of hyperparams
def iterate_params(current : list[int], max_hyper: list[int]) -> list[int]:
    for i, (max, curr) in enumerate(zip(max_hyper, current)):
        if max == curr:
            current[i] = 0
        else:
            current[i] += 1
            break
    return current

# create dictionary of parameters given the chosen ones
def choose_params(current: list[int], hyperparams: dict[str, list[int | str]]) -> dict[str, int | str]:
    hyper = {}
    for hyper_nb, (hyper_name, hyper_choices) in zip(current, hyperparams.items()):
        hyper[hyper_name] = hyper_choices[hyper_nb]
    return hyper

# nb of possibility for all hyperparams
def nb_possibility(max_hyper: list[int]) -> int:
    total = 1
    for nb_poss in max_hyper:
        total *= (nb_poss + 1)
    return total

# training the model with given hyperparams
def train_hyper(hyperparams: dict[str, list[int | str]], model: Model, X: np.ndarray, y: np.ndarray, split: int) -> tuple[Model, float, dict[str, int|str], dict[tuple[int|str], float]]:
    scores = {}

    # params choice
    current = [0 for _ in range(len(hyperparams))]
    max_hyper = [len(hyperparam) - 1 for hyperparam in hyperparams.values()]
    current_params = choose_params(current, hyperparams)
    all_poss = nb_possibility(max_hyper)

    # training model
    trained_model = model(**current_params)

    # register score
    best_score = cross_val_score(trained_model, X, y, cv=split).mean()
    best_params = current_params.copy()
    best_model = deepcopy(trained_model)
    scores[tuple([param for param in current_params.values()])] = deepcopy(best_score)

    print(all_poss)
    i = 0
    nb_print = (all_poss//4) + 1
    while not all(np.equal(current, max_hyper)):
        # choose params
        current = iterate_params(current, max_hyper)
        current_params = choose_params(current, hyperparams)

        # train model
        trained_model = model(**current_params)
        current_score = cross_val_score(trained_model, X, y, cv=split).mean()
        scores[tuple([param for param in current_params.values()])] = deepcopy(current_score)

        # update best if better score
        if current_score > best_score:
            best_score = deepcopy(current_score)
            best_params = deepcopy(current_params)
            best_model = deepcopy(trained_model)

        # print params every now and then
        if i % nb_print == 0:
            print(i, current_params)
        i += 1
    return (best_model, best_score, best_params, scores)