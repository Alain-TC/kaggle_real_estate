from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder
from .transformers.fillna_transformer import FillnaMeanTransformer
from .transformers.normalize_transformer import NormalizeTransformer
from .transformers.add_column_transformer import CreateTotalSFTransformer
from .transformers.box_cox_transformer import BoxCoxTransformer
from .transformers.column_selector_transformer import ExcludeColumnsTransformer
from .transformers.onehot_encoder_transformer import SimpleOneHotEncoder


def remove_outliers(data, columns_config):
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]

    pipeline = make_pipeline(ExcludeColumnsTransformer(["Id"]),
                             CreateTotalSFTransformer(),
                             BoxCoxTransformer(quantitative_columns),
                             FillnaMeanTransformer(quantitative_columns),
                             TargetEncoder(semi_quali_columns),
                             SimpleOneHotEncoder(qualitative_columns),
                             NormalizeTransformer(quantitative_columns)
                             )

    # Prepare Data Training
    X = data
    y = data[['SalePrice']]
    X = pipeline.fit_transform(X, y)

    # fit the model
    clf = IsolationForest(max_samples=100)
    clf.fit(X)
    outlier_index = clf.predict(X)

    return data[outlier_index == 1]
