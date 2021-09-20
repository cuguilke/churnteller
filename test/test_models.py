import unittest

from models import get_parameters

class TestModels(unittest.TestCase):
    def test_get_parameters(self):
        parameters = get_parameters("xgboost", use_best=False)
        self.assertIn("max_depth", parameters)
        self.assertIn("n_estimators", parameters)
        self.assertIn("learning_rate", parameters)

if __name__ == '__main__':
    unittest.main()