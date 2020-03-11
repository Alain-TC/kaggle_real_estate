import numpy as np


def transform_log(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: np.log(x))
    return df


def transform_exp(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: np.exp(x))
    return df
