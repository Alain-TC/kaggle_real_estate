import os
import pandas as pd


def make_dir_if_not_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def export_dataframe_csv(df, path):
    make_dir_if_not_exists(path)
    df.to_csv(path, index=False)


def csv_to_df(input_file, sep=',', names=None, header=None):
    return pd.read_table(input_file, sep=sep, header=header, names=names)
