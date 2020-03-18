from sklearn.base import TransformerMixin
import numpy

class ShowTransformer(TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):

        print(self.name)
        #print(set(df["Street"]))
        print("df")
        #print(df)
        print("null")
        try:
            print("SHOW NAN2?")
            print(df.isnull().sum().sum())
            print(max(df))
            print(min(df))
            print("SHOW NAN2")
            df.to_csv('totototo.csv', index=False)
        except:
            print("SHOW NAN1?")
            print(numpy.isnan(df).any())
            print(numpy.max(df))
            print(numpy.min(df))
            print(numpy.isnan(df).any())
            print("SHOW NAN1")
            #pass
        try:
            print("df.PoolArea")
            print(df.PoolArea)
        except:
            pass

        return df
