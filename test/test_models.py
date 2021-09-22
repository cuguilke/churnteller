import os
import json
import unittest

from models import get_parameters, save_parameters

class TestModels(unittest.TestCase):
    def test_get_parameters(self):
        parameters = get_parameters("xgboost", "RFM", False, use_best=False)
        self.assertIn("max_depth", parameters)
        self.assertIn("n_estimators", parameters)
        self.assertIn("learning_rate", parameters)

        parameters = get_parameters("svm", "RFM", False, use_best=False)
        self.assertIn("C", parameters)
        self.assertIn("gamma", parameters)
        self.assertIn("kernel", parameters)
        self.assertIn("max_iter", parameters)

    def test_save_parameters(self):
        dummy_parameter_dict = {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}

        # Create a dummy file with existing param record
        with open("dummy.json", "w+") as dummy_file:
            json.dump({"xgboost:RFM": dummy_parameter_dict}, dummy_file)

        with self.assertWarns(RuntimeWarning):
            save_parameters("xgboost", "RFM", False, dummy_parameter_dict, path="dummy.json")

        # Delete the dummy file
        os.remove("dummy.json")

if __name__ == '__main__':
    unittest.main()