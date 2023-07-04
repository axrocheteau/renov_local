# librairies
from lib.usefull import *
import numpy as np
import seaborn as sn
import sklearn as sk
import pyspark as ps
import pandas as pd
import matplotlib as matplot
import matplotlib.pyplot as plt

# score
from sklearn.metrics import confusion_matrix

# linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

# random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# XGboost
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier

# typing
Dataframe = ps.sql.dataframe.DataFrame
Axe = matplot.axes._axes.Axes

Model = XGBRegressor | XGBClassifier | RandomForestClassifier | RandomForestRegressor | Ridge | LogisticRegression


# plot repartition of variables in dataset


def get_percent(value: float, values: list[float]) -> str:
    '''return labels for pie'''
    total = sum(values)
    if len(values) < 5:
        return f'{value/100*total:.0f}\n{value:.2f}%'
    else:
        return f'{value:.1f}'


def plot_repartition(df: Dataframe, dictionary: Dataframe, variable: str, ax: Axe, title: str = None) -> None:
    '''plot pie'''
    labels = get_labels(dictionary, variable)['meaning']
    count = get_dict(df.withColumn(variable, F.col(variable).cast(
        int)).groupBy(variable).count().orderBy(variable))['count']
    ax.pie(count, labels=labels, autopct=lambda x: get_percent(
        x, count), startangle=90)
    if title:
        ax.set_title(title)


def plot_hist(df: Dataframe, dictionary: Dataframe, variable: str, ax: Axe, title: str = None) -> None:
    '''plot_hist'''
    labels = get_labels(dictionary, variable)['meaning']
    values = get_dict(df.groupBy(variable).count().orderBy(variable))
    total = sum(values['count'])
    percent = [value * 100 / total for value in values['count']]
    x = values[variable]
    min_value = int(min(values[variable]))

    # bar graph
    bars = ax.bar(x=x, height=percent, width=0.5)
    ax.set_xticks(range(min_value, len(labels) +
                  min_value), labels, rotation=90)

    # set y ticks
    vals = ax.get_yticks()
    ax.set_yticklabels(['%1.2f%%' % i for i in vals])

    # plot percentage
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", (bar.get_x() + bar.get_width()/2,
                    height+.05), ha="center", va="bottom", fontsize=10)

    # title
    if title:
        ax.set_title(title)


def compare_repartition(dfs: list[Dataframe], dictionary: Dataframe, variable: str) -> None:
    '''plot multiple chart'''
    f, ax = plt.subplots(1, len(dfs), figsize=(20, 5), sharey=True)
    plt.subplots_adjust(wspace=0.2)
    plt.suptitle(variable)
    if len(dfs) > 1:
        for i, df in enumerate(dfs):
            plot_hist(df, dictionary, variable, ax[i], retrieve_name(df))
    else:
        plot_hist(dfs[0], dictionary, variable, ax, retrieve_name(dfs[0]))


def histo_continuous(df: Dataframe, variable: str, bins: int = 20) -> None:
    '''plot histogram'''
    plt.title(variable)
    plt.hist(df.select(variable).toPandas(), bins=bins)
    plt.show()

# show ML results


def show_hyperparam_opti(scores: dict[tuple[int | str], float], hyperparams: dict[str, list[int | str]], ax: Axe, model_name: str) -> None:
    '''plot sorted scores for each hyperparams choice'''
    x_pos = [i for i in range(len(scores))]
    scores = dict(sorted(scores.items(), key=lambda x: x[1]))
    ax.plot(x_pos, scores.values())
    ax.set_xticks(x_pos, labels=scores.keys(), rotation=90)
    ax.set_title(f'''{model_name}\n{tuple(hyperparams.keys())}''')


def show_result(y_pred: np.ndarray, y_true: np.ndarray, ax: Axe, model_name: str) -> None:
    '''show prediction against predicted values (continuous)'''
    df = pd.DataFrame(np.column_stack((y_pred, y_true)),
                      columns=['pred', 'true'])
    df = df.sort_values(by=['true'])
    df.reset_index(inplace=True, drop=True)
    ax.plot(df['true'])
    ax.scatter(df['pred'])
    ax.legend(['pred', 'true'])
    ax.set_ylabel('surface')
    ax.set_xticks([])
    ax.set_title(model_name)


def show_matrix(y_pred: np.ndarray, y_true: np.ndarray, ax: Axe, model_name: str) -> None:
    '''confusion matrix (result for categorical values)'''
    matrix = confusion_matrix(y_true, y_pred)
    sn.heatmap(matrix/np.sum(matrix), ax=ax, annot=True, fmt='.1%')
    ax.set_title(model_name)


def show_importance(model: Model, labels: list[int], ax: Axe, model_name: str) -> None:
    '''show importance of variables for the model'''
    y_pos = [i for i in range(len(labels))]
    if isinstance(model, Ridge):
        ax.barh(y_pos, model.coef_.T.ravel(), align='center')
    elif isinstance(model, LogisticRegression):
        ax.barh(y_pos, model.coef_.T.mean(axis=1).ravel(), align='center')

    elif isinstance(model, RandomForestRegressor) or isinstance(model, RandomForestClassifier) or isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
        ax.barh(y_pos, model.feature_importances_.T.ravel(), align='center')
    else:
        print('pol degree > 1, too many features')
    ax.set_yticks(y_pos, labels=labels)
    ax.set_title(model_name)
