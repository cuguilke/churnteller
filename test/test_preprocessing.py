import os
import unittest

from preprocessing import load_data

class TestPreprocessing(unittest.TestCase):
    def test_load_data(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.assertTrue(os.path.join(ROOT_DIR, "../data/machine_learning_challenge_order_data.csv.gz"), msg="Training data is not found!")
        self.assertTrue(os.path.join(ROOT_DIR, "../data/machine_learning_challenge_labeled_data.csv.gz"), msg="Test data is not found!")

        customer_info = load_data()
        for customer in customer_info:
            self.assertIsNotNone(customer_info[customer]["label"])


if __name__ == '__main__':
    unittest.main()