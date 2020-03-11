

def split_dataframe_by_row(df, ratio):
    df = df.sample(frac=1)
    df1 = df.iloc[:int(df.shape[0] * ratio), :]
    df2 = df.iloc[int(df.shape[0] * ratio):, :]
    return df1, df2
