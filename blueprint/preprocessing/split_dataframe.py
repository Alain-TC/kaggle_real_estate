

def split_dataframe_by_row(df, ratio, seed=42):
    df = df.sample(frac=1, random_state=seed)
    print("TEST")
    print(df.shape[0])
    print(df.shape[0] * ratio)
    print(int(df.shape[0] * ratio))
    df1 = df.iloc[:int(df.shape[0] * ratio), :].reset_index(inplace=False, drop=True)
    df2 = df.iloc[int(df.shape[0] * ratio):, :].reset_index(inplace=False, drop=True)
    return df1, df2
