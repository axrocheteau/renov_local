import numpy as np
from copy import deepcopy

# score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

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

# knn
from sklearn.impute import KNNImputer

from lib.prepare_data import get_predict_set

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


def iterate_params(current: list[int], max_hyper: list[int]) -> list[int]:
    """get next coverall of hyperparams"""
    for i, (max, curr) in enumerate(zip(max_hyper, current)):
        if max == curr:
            current[i] = 0
        else:
            current[i] += 1
            break
    return current


def choose_params(
    current: list[int], hyperparams: dict[str, list[int | str]]
) -> dict[str, int | str]:
    """create dictionary of parameters given the chosen ones"""
    hyper = {}
    for hyper_nb, (hyper_name, hyper_choices) in zip(current, hyperparams.items()):
        hyper[hyper_name] = hyper_choices[hyper_nb]
    return hyper


def nb_possibility(max_hyper: list[int]) -> int:
    """nb of possibility for all hyperparams"""
    total = 1
    for nb_poss in max_hyper:
        total *= nb_poss + 1
    return total


def score_inputer(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[float], float]:
    scores = []
    for i in range(y_pred.shape[1]):
        scores.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    return (scores, np.mean(scores))


def train_hyper(
    hyperparams: dict[str, list[int | str]],
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    split: ShuffleSplit,
    random_state: int,
    categorical_feature: list[int],
    verbose: int,
    scoring: str,
) -> tuple[Model, float, dict[str, int | str], dict[tuple[int | str], float]]:
    """training the model with given hyperparams"""
    if isinstance(model, HistGradientBoostingRegressor) or isinstance(
        model, HistGradientBoostingClassifier
    ):
        search = HalvingRandomSearchCV(
            model(categorical_features=categorical_feature),
            param_distributions=hyperparams,
            min_resources=100,
            random_state=random_state,
            cv=split,
            scoring=scoring,
            verbose=0,
            error_score=0,
        )
    else:
        search = HalvingRandomSearchCV(
            model(),
            param_distributions=hyperparams,
            min_resources=100,
            random_state=random_state,
            cv=split,
            scoring=scoring,
            verbose=verbose,
            error_score=0,
        )
    search.fit(X, y)
    scores = {}
    for params, score in zip(
        search.cv_results_["params"], search.cv_results_["mean_test_score"]
    ):
        scores[tuple([param_value for param_value in params.values()])] = score
    best_model = search.best_estimator_
    best_score = search.best_score_
    best_params = search.best_params_
    return (best_model, best_score, best_params, scores)


def gridsearch_inputer(
    model: KNNImputer,
    hyperparams: dict[str, list[int | str]],
    training: np.ndarray,
    truth: np.ndarray,
    split: list[np.ndarray],
) -> tuple[dict[tuple[str | int | float], float], float, list[str | int | float]]:
    current = [0 for _ in range(len(hyperparams))]
    max_hyper = [len(cut_param) - 1 for cut_param in hyperparams.values()]
    current_params = choose_params(current, hyperparams)
    all_poss = nb_possibility(max_hyper)
    print(all_poss)
    nb_print = (all_poss // 3) + 1

    trained_model = model(**current_params)
    pred = trained_model.fit_transform(training)

    y_true = get_predict_set(truth, split)
    y_pred = np.rint(get_predict_set(pred, split))

    scores_current, score = score_inputer(y_true, y_pred)

    max_score = deepcopy(score)
    best_params = deepcopy(current_params)
    scores = {}
    scores[tuple([param for param in current_params.values()])] = deepcopy(score)

    i = 1
    while not all(np.equal(current, max_hyper)):
        # choose params
        current = iterate_params(current, max_hyper)
        current_params = choose_params(current, hyperparams)
        if i % nb_print == 0:
            print(i)
        i += 1

        trained_model = model(**current_params)
        pred = trained_model.fit_transform(training)
        y_pred = np.rint(get_predict_set(pred, split))

        scores_current, score = score_inputer(y_true, y_pred)
        scores[tuple([param for param in current_params.values()])] = deepcopy(score)
        if score > max_score:
            max_score = deepcopy(score)
            best_params = deepcopy(current_params)

    print(max_score, best_params)
    return scores, max_score, best_params
