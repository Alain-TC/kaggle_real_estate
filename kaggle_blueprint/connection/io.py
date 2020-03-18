import os
import json
import numpy
import pandas as pd


def convert(o):
    if isinstance(o, numpy.int64):
        return int(o)
    raise TypeError


def make_dir_if_not_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def export_dataframe_csv(df, path):
    make_dir_if_not_exists(path)
    df.to_csv(path, index=False)


def csv_to_df(input_file, sep=',', names=None, header=None):
    return pd.read_table(input_file, sep=sep, header=header, names=names)


def write_json(item, path):
    with open(path, 'w') as json_file:
        json_file.write(json.dumps(item, default=convert))


def read_json(path):
    with open(path) as json_file:
        item = json.load(json_file)
    return item
