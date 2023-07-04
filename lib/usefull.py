# librairies
import sklearn as sk
from pyspark.sql import functions as F
import pyspark as ps
import matplotlib.pyplot as plt
import matplotlib as matplot
import inspect

Dataframe = ps.sql.dataframe.DataFrame


def retrieve_name(var: str) -> list[str]:
    '''get the name of  variable outside fonction'''
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def cut(name: str) -> str:
    '''cut variable name if too long'''
    if isinstance(name, str) and len(name) > 25:
        return name[0:25]
    return name


def get_dict(df: Dataframe) -> dict[str, list[str]]:
    '''get all values in dict form'''
    col_names = [col[0] for col in df.dtypes]
    values = {}
    rows = df.collect()
    for col_name in col_names:
        values[col_name] = [cut(row[col_name]) for row in rows]
    return values


def get_labels(dictionary: Dataframe, variable: str) -> dict[str, list[str]]:
    '''get labels for a variable thanks to dictionary'''
    study = (
        dictionary.filter(
            (F.col('column') == variable) &
            (~F.col('value').contains('-'))
        )
        .select(
            F.col('value'),
            F.col('meaning')
        )
        .dropDuplicates()
        .orderBy(F.col('value').cast('int'))
    )
    return get_dict(study)
