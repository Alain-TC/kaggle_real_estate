from category_encoders import TargetEncoder
from sklearn.pipeline import make_pipeline
from ..preprocessing.transformers.add_column_transformer import CreateSumTransformer, CreateOneHotTransformer, \
    NewHouseTransformer
from ..preprocessing.transformers.box_cox_transformer import BoxCoxTransformer
from ..preprocessing.transformers.fillna_transformer import FillnaMeanTransformer, FillnaMeanMatrixTransformer
from ..preprocessing.transformers.normalize_transformer import NormalizeTransformer
from ..preprocessing.transformers.onehot_encoder_transformer import SimpleOneHotEncoder
from ..preprocessing.transformers.column_selector_transformer import ExcludeColumnsTransformer
from sklearn.feature_selection import SelectKBest, chi2


# pipeline de nettoyage des données, utilisé avant la modélisation
def pipe_preprocessing(columns_config):
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]
    target_features_list = columns_config["target_features_list"]
    indicator_features_list = columns_config["indicator_features_list"]
    all_qualitative_columns = qualitative_columns + semi_quali_columns

    preprocessing_pipeline = make_pipeline(ExcludeColumnsTransformer(["Id"]),
                                           CreateSumTransformer(target_features_list),
                                           CreateOneHotTransformer(indicator_features_list),
                                           NewHouseTransformer(),
                                           BoxCoxTransformer(quantitative_columns)  # ,
                                           # SelectKBest(chi2, k=60)
                                           )
    return preprocessing_pipeline


# pipeline de feature engineering, utilisé pendant la modélisation
def pipe_processing(columns_config):
    quantitative_columns = columns_config["quantitative_columns"]
    semi_quali_columns = columns_config["semi_quali_columns"]
    qualitative_columns = columns_config["qualitative_columns"]

    processing_pipeline = make_pipeline(FillnaMeanTransformer(quantitative_columns),
                                        NormalizeTransformer(quantitative_columns),
                                        TargetEncoder(semi_quali_columns),
                                        SimpleOneHotEncoder(qualitative_columns),
                                        FillnaMeanMatrixTransformer()
                                        )
    return processing_pipeline
