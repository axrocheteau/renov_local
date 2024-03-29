# librairies
import numpy as np
import seaborn as sn
import pyspark as ps
import pandas as pd
import matplotlib as matplot
import matplotlib.pyplot as plt
from typing import Callable
from pyspark.sql import functions as F

# import other functions
from lib.useful import get_labels, get_dict, retrieve_name

# score
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, f1_score
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm

# linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

# random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# XGboost
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier

# quick XGboost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

# typing
Dataframe = ps.sql.dataframe.DataFrame
Axe = matplot.axes._axes.Axes
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


# plot repartition of variables in dataset


def get_percent(value: float, values: list[float]) -> str:
    """return labels for pie"""
    total = sum(values)
    if len(values) < 5:
        return f"{value/100*total:.0f}\n{value:.2f}%"
    else:
        return f"{value:.1f}"


def plot_repartition(
    df: Dataframe, dictionary: Dataframe, variable: str, ax: Axe, title: str = None
) -> None:
    """plot pie"""
    labels = get_labels(dictionary, variable)["meaning"]
    count = get_dict(
        df.withColumn(variable, F.col(variable).cast(int))
        .groupBy(variable)
        .count()
        .orderBy(variable)
    )["count"]
    ax.pie(count, labels=labels, autopct=lambda x: get_percent(x, count), startangle=90)
    if title:
        ax.set_title(title)


def plot_hist(
    df: Dataframe, dictionary: Dataframe, variable: str, ax: Axe, title: str = None
) -> None:
    """plot_hist"""
    labels = get_labels(dictionary, variable)["meaning"]
    values = get_dict(df.groupBy(variable).count().orderBy(variable))
    total = sum(values["count"])
    percent = [value * 100 / total for value in values["count"]]
    x = values[variable]
    min_value = int(min(values[variable]))

    # bar graph
    bars = ax.bar(x=x, height=percent, width=0.5)
    ax.set_xticks(range(min_value, len(labels) + min_value), labels, rotation=90)

    # set y ticks
    vals = ax.get_yticks()
    ax.set_yticklabels(["%1.2f%%" % i for i in vals])

    # plot percentage
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height + 0.05),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # title
    if title:
        ax.set_title(title)


def compare_repartition(
    dfs: list[Dataframe], dictionary: Dataframe, variable: str
) -> None:
    """plot multiple chart"""
    f, ax = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 5), sharey=True)
    plt.subplots_adjust(wspace=0.2)
    plt.suptitle(variable)
    if len(dfs) > 1:
        for i, df in enumerate(dfs):
            plot_hist(df, dictionary, variable, ax[i], retrieve_name(df))
    else:
        plot_hist(dfs[0], dictionary, variable, ax, retrieve_name(dfs[0]))


def histo_continuous(df: Dataframe, variable: str, bins: int = 20) -> None:
    """plot histogram"""
    plt.title(variable)
    plt.hist(df.select(variable).toPandas(), bins=bins)
    plt.show()


# show ML results


def show_hyperparam_opti(
    scores: dict[tuple[int | str], float],
    hyperparams: dict[str, list[int | str]],
    ax: Axe,
    model_name: str,
) -> None:
    """plot sorted scores for each hyperparams choice"""
    x_pos = [i for i in range(len(scores))]
    scores = dict(sorted(scores.items(), key=lambda x: x[1]))
    ax.plot(x_pos, scores.values())
    ax.set_xticks(x_pos, labels=scores.keys(), rotation=90)
    ax.set_title(f"""{model_name}\n{tuple(hyperparams.keys())}""")


def show_result(
    y_pred: np.ndarray, y_true: np.ndarray, ax: Axe, model_name: str
) -> None:
    """show prediction against predicted values (continuous)"""
    df = pd.DataFrame(np.column_stack((y_pred, y_true)), columns=["pred", "true"])
    df = df.sort_values(by=["true"])
    df.reset_index(inplace=True, drop=True)
    scale = max(df.index) / max(df["true"])
    heatmap, xedges, yedges = np.histogram2d(df.index / scale, df["pred"], bins=150)
    heatmap = gaussian_filter(heatmap, sigma=20)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin="lower", cmap=cm.jet)
    ax.plot(df.index / scale, df["true"], c="black")
    ax.legend(["true"])
    ax.set_xticks([])
    ax.set_title(f"{model_name}\n score : {round(r2_score(y_true, y_pred),4)}")


def show_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ax: Axe,
    model_name: str,
    labels: list[str] = None,
    score: str | Callable = None,
) -> None:
    """confusion matrix (result for categorical values)"""
    matrix = confusion_matrix(y_true, y_pred)
    if labels:
        sn.heatmap(
            (matrix.T / np.sum(matrix, axis=1).T).T,
            ax=ax,
            annot=True,
            fmt=".1%",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        sn.heatmap(
            (matrix.T / np.sum(matrix, axis=1).T).T, ax=ax, annot=True, fmt=".1%"
        )
    ax.set_ylabel("true")
    ax.set_xlabel("pred")
    if score is None or isinstance(score, str):
        ax.set_title(
            f'{model_name}\nscore : {round(f1_score(y_true, y_pred, average="micro"),4)}'
        )
    else:
        ax.set_title(f"{model_name}\nscore : {round(score(y_true, y_pred),4)}")


def show_importance(model: Model, labels: list[str], ax: Axe, model_name: str) -> None:
    """show importance of variables for the model"""
    y_pos = [i for i in range(len(labels))]
    if isinstance(model, Ridge):
        ax.barh(y_pos, model.coef_.T.ravel(), align="center")
    elif isinstance(model, LogisticRegression):
        ax.barh(y_pos, model.coef_.T.mean(axis=1).ravel(), align="center")

    elif (
        isinstance(model, RandomForestRegressor)
        or isinstance(model, RandomForestClassifier)
        or isinstance(model, XGBClassifier)
        or isinstance(model, XGBRegressor)
        or isinstance(model, DecisionTreeClassifier)
    ):
        ax.barh(y_pos, model.feature_importances_.T.ravel(), align="center")

    else:
        print("not handled for this type of algorithm")
    ax.set_yticks(y_pos, labels=labels)
    ax.set_title(model_name)


def score_plot(y_pred: np.ndarray, y_true: np.ndarray, varnames: list[str]):
    """multivariate inputation"""
    f, ax_result = plt.subplots(1, y_pred.shape[1], figsize=(15, 5))
    for i in range(y_pred.shape[1]):
        if len(np.unique(y_true[:, i])) > 10:
            score = 1 - (
                np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
                / np.sum((np.mean(y_true[:, i]) - y_true[:, i]) ** 2)
            )
            show_result(
                y_pred[:, i],
                y_true[:, i],
                ax_result[i],
                varnames[i] + f"\n score : {round(score,2)}",
            )
        else:
            score = accuracy_score(y_true[:, i], y_pred[:, i])
            show_matrix(
                y_pred[:, i].ravel(),
                y_true[:, i].ravel(),
                ax_result[i],
                varnames[i] + f"\n score : {round(score,2)}",
            )
