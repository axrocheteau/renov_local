# librairies
import numpy as np
from pyspark.sql import functions as F
import pyspark as ps
import pandas as pd

# prepare data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from copy import copy

# import other functions
from useful import *

Dataframe = ps.sql.dataframe.DataFrame


def prepare_dataset(df: Dataframe, dictionary: Dataframe, col_X_hot: list[str], col_X_not_hot: list[str], col_y: list[str], y_hot: bool, scale: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    '''prepare data for ML algo'''
    # get labels of variables to plot importance
    labels = ['dummy']
    for variable in (col_X_not_hot):
        labels.append(variable)
    for variable in col_X_hot:
        labels.extend(get_labels(dictionary, variable)['meaning'][1:])

    # prepare X data with one hot encoder if necessary
    X_not_hot = df.select(col_X_not_hot).toPandas().to_numpy()
    if len(col_X_hot) > 0:
        X_hot = df.select(col_X_hot).toPandas()
        X_hot = OneHotEncoder(
            drop='first', sparse_output=False).fit_transform(X_hot)
        X = np.column_stack((X_not_hot, X_hot))
    else:
        X = X_not_hot.copy()

    # prepare y data
    y = df.select(col_y).toPandas().to_numpy()
    if y_hot:
        y = OneHotEncoder(drop='first', sparse_output=False).fit_transform(y)
    else:
        y = y.ravel()
        if 0 not in pd.unique(y):
            y = y - 1
    
    # limit number of training_data
    if X.shape[0] > 100000:
        np.random.seed(42)
        indexes = np.random.choice(X.shape[0], 100000, replace=False)
        X = X[indexes, :]
        y = y[indexes]


    # scale input data
    if scale:
        X = StandardScaler().fit_transform(X)
    return (X, y, labels)


def get_coef(rows: ps.sql.Row, sum: int) -> float:
    ''' get the coef according to data repartition to correct score'''
    return (rows[-1]['count']/sum) - (rows[0]['count']/sum)


def make_cut(first_value: int, steps: int, nb_steps: int) -> list[int]:
    ''' return the cuts to make on data'''
    cuts = [first_value] + [first_value +
                            steps * (i+1) for i in range(nb_steps)]
    return cuts


def to_categorical(df: Dataframe, variable: str, cuts: list[int]) -> tuple[Dataframe, float]:
    '''change a variable in a dataframe to categorical with defined cuts'''
    request = 'CASE '  # use sql to build request
    for i, cut in enumerate(cuts):
        if i == 0:  # first instruction
            request += f"WHEN {variable} < {cut} THEN {i}\n"
        else:  # middle instructions
            request += f"WHEN {variable} >= {last_cut} AND {variable} < {cut} THEN {i}\n"
        last_cut = copy(cut)
    # last instruction
    request += f"WHEN surface >= {last_cut} THEN {len(cuts)}\nEND"

    # dataframe that needs to be changed
    study = df.withColumn('surface', F.expr(request))
    rows_study = study.groupBy(variable).count().orderBy('count').collect()
    sum_study = study.count()

    # moderate coef 2 times housing 1 time dpe (prediction is happening on housing)
    coef = 1 - (get_coef(rows_study, sum_study))
    return (study, coef)

def get_predict_set(complete_df: np.ndarray, split: list[np.ndarray]):
    '''multivariate inputation'''
    y=np.array([])
    for i, (_, test_index) in enumerate(split):
        if i == 0:
            y = complete_df[test_index,-(i+1)]
        else:
            y = np.column_stack((y, complete_df[test_index,-(i+1)]))
    return y
