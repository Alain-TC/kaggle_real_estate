import unittest
from kaggle_blueprint.modelisation.config_hyperopt import get_config_hyperopt


class Testget_config_hyperopt(unittest.TestCase):
    def setUp(self):
        self.key_set = {"model__alpha_1", "model__alpha_2", "model__lambda_1", "model__lambda_2", "model__n_iter",
                        "model__normalize"}

    def test_transform(self):
        columns_config = get_config_hyperopt("BayesianRidge")
        self.assertEqual(columns_config.keys(), self.key_set)
