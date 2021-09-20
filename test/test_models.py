import os
import json
import unittest

from models import get_parameters, save_parameters

class TestModels(unittest.TestCase):
    def test_get_parameters(self):
        parameters = get_parameters("xgboost", use_best=False)
        self.assertIn("max_depth", parameters)
        self.assertIn("n_estimators", parameters)
        self.assertIn("learning_rate", parameters)

    def test_save_parameters(self):
        dummy_parameter_dict = {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}

        # Create a dummy file with existing param record
        with open("dummy.json", "w+") as dummy_file:
            json.dump({"xgboost": dummy_parameter_dict}, dummy_file)

        with self.assertWarns(RuntimeWarning):
            save_parameters("xgboost", dummy_parameter_dict, path="dummy.json")

        # Delete the dummy file
        os.remove("dummy.json")

if __name__ == '__main__':
    unittest.main()