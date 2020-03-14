from sklearn.pipeline import make_pipeline
from category_encoders import TargetEncoder

from preprocessing.transformers.fillna_transformer import FillnaMeanTransformer, FillnaMeanMatrixTransformer
from preprocessing.transformers.normalize_transformer import NormalizeTransformer
from preprocessing.transformers.add_column_transformer import CreateTotalSFTransformer
from preprocessing.transformers.box_cox_transformer import BoxCoxTransformer
from preprocessing.transformers.column_selector_transformer import ExcludeColumnsTransformer
from preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder


# pipeline de nettoyage des données, utilisé avant la modélisation
def pipe_preprocessing(columns_config):
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]

    all_qualitative_columns = qualitative_columns + semi_quali_columns
    return (make_pipeline(ExcludeColumnsTransformer(["Id"]),
                          CreateTotalSFTransformer(),
                          BoxCoxTransformer(quantitative_columns)))


# pipeline de feature engineering, utilisé pendant la modélisation
def pipe_processing(columns_config):
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]
    return (make_pipeline(
        FillnaMeanTransformer(quantitative_columns),
        NormalizeTransformer(quantitative_columns),
        TargetEncoder(semi_quali_columns),
        SimpleOneHotEncoder(qualitative_columns),
        FillnaMeanMatrixTransformer()))
