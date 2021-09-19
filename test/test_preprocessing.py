import os
import unittest
import numpy as np

from preprocessing import load_data, get_train_val_split, get_RFM_data

class TestPreprocessing(unittest.TestCase):
    def test_load_data(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        order_data_path = "./data/machine_learning_challenge_order_data.csv.gz"
        label_data_path = "./data/machine_learning_challenge_labeled_data.csv.gz"

        self.assertTrue(os.path.join(ROOT_DIR, order_data_path), msg="Training data is not found!")
        self.assertTrue(os.path.join(ROOT_DIR, label_data_path), msg="Test data is not found!")

        customer_info = load_data(order_data_path, label_data_path)
        for customer in customer_info:
            self.assertIsNotNone(customer_info[customer]["label"])

    def test_get_train_val_split(self):
        dummy_x = np.arange(10)
        dummy_y = np.array([1, 1, 0, 0 ,0 ,0 ,0 ,0, 0, 0])

        (x_train, y_train), (x_val, y_val) = get_train_val_split(dummy_x, dummy_y, val_split=0.5, n_splits=1)[0]

        train_has_1 = False
        for y in y_train:
            if y == 1:
                train_has_1 = True

        val_has_1 = False
        for y in y_val:
            if y == 1:
                val_has_1 = True

        self.assertTrue(train_has_1 and val_has_1, msg="Train/val split inbalance!")

    def test_get_RFM_data(self):
        dummy_customer_info = {
            "0f9581390584": {'label': 1, 'data': [{'order_date': '2016-06-06', 'order_hour': 17, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 5.31, 'restaurant_id': 53403498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4324}, {'order_date': '2016-06-13', 'order_hour': 16, 'customer_order_rank': 2.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 6.903, 'restaurant_id': 53383498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2016-06-20', 'order_hour': 12, 'customer_order_rank': 3.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 6.903, 'restaurant_id': 53383498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2016-07-04', 'order_hour': 17, 'customer_order_rank': 4.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 6.903, 'restaurant_id': 53383498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4324}, {'order_date': '2016-07-12', 'order_hour': 16, 'customer_order_rank': 5.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 6.903, 'restaurant_id': 53383498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2017-02-05', 'order_hour': 20, 'customer_order_rank': 6.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 9.6642, 'restaurant_id': 53383498, 'city_id': 70404, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}]},
            "0f959b6d1cae": {'label': 0, 'data': [{'order_date': '2016-11-01', 'order_hour': 20, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 1.715, 'delivery_fee': 0.493, 'amount_paid': 6.2658, 'restaurant_id': 175053498, 'city_id': 33833, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 4356}]},
            "0f93abf0f99c": {'label': 1, 'data': [{'order_date': '2017-01-08', 'order_hour': 17, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 7.2747, 'restaurant_id': 252483498, 'city_id': 51602, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2017-01-11', 'order_hour': 20, 'customer_order_rank': 2.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 5.2569, 'restaurant_id': 292523498, 'city_id': 51602, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4228}, {'order_date': '2017-01-14', 'order_hour': 22, 'customer_order_rank': 3.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 7.90128, 'restaurant_id': 320643498, 'city_id': 51602, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4228}, {'order_date': '2017-01-20', 'order_hour': 22, 'customer_order_rank': None, 'is_failed': 1, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 6.5844, 'restaurant_id': 287463498, 'city_id': 51602, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 212}, {'order_date': '2017-02-05', 'order_hour': 19, 'customer_order_rank': 4.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 5.31, 'restaurant_id': 287463498, 'city_id': 51602, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2017-02-15', 'order_hour': 17, 'customer_order_rank': 5.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 10.8855, 'restaurant_id': 205763498, 'city_id': 51602, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4356}]},
            "19893650265d": {'label': 0, 'data': [{'order_date': '2016-12-22', 'order_hour': 17, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 9.5049, 'restaurant_id': 64093498, 'city_id': 84259, 'payment_id': 1619, 'platform_id': 29463, 'transmission_id': 4324}]},
            "198960bb3a18": {'label': 0, 'data': [{'order_date': '2017-01-29', 'order_hour': 16, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 9.2925, 'restaurant_id': 83503498, 'city_id': 94263, 'payment_id': 1491, 'platform_id': 30359, 'transmission_id': 4324}]},
            "1989c7ba7ff9": {'label': 0, 'data': [{'order_date': '2015-06-06', 'order_hour': 19, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 16.0362, 'restaurant_id': 154773498, 'city_id': 73500, 'payment_id': 1619, 'platform_id': 30359, 'transmission_id': 4228}]},
            "198b691553d4": {'label': 0, 'data': [{'order_date': '2016-04-15', 'order_hour': 18, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 10.5138, 'restaurant_id': 106063498, 'city_id': 15748, 'payment_id': 1619, 'platform_id': 30231, 'transmission_id': 4356}]},
            "198b6e7139df": {'label': 0, 'data': [{'order_date': '2015-06-16', 'order_hour': 14, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 5.3631, 'restaurant_id': 164753498, 'city_id': 10346, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 4228}, {'order_date': '2015-06-30', 'order_hour': 14, 'customer_order_rank': 2.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 8.6022, 'restaurant_id': 51693498, 'city_id': 10346, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2015-09-17', 'order_hour': 19, 'customer_order_rank': 3.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 5.44806, 'restaurant_id': 117483498, 'city_id': 10346, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 4228}]},
            "198b8b884a5b": {'label': 1, 'data': [{'order_date': '2015-09-13', 'order_hour': 20, 'customer_order_rank': 1.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 8.496, 'restaurant_id': 187233498, 'city_id': 72358, 'payment_id': 1779, 'platform_id': 30231, 'transmission_id': 4324}, {'order_date': '2015-09-27', 'order_hour': 19, 'customer_order_rank': 2.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 11.151, 'restaurant_id': 187233498, 'city_id': 72358, 'payment_id': 1779, 'platform_id': 29463, 'transmission_id': 4356}, {'order_date': '2015-10-28', 'order_hour': 17, 'customer_order_rank': 3.0, 'is_failed': 0, 'voucher_amount': 0.0, 'delivery_fee': 0.0, 'amount_paid': 11.151, 'restaurant_id': 187233498, 'city_id': 72358, 'payment_id': 1779, 'platform_id': 30231, 'transmission_id': 4356}]}
        }

        x, y = get_RFM_data(dummy_customer_info)
        self.assertEqual(x.shape[0], y.shape[0], msg="Input - output mismatch!")
        self.assertTrue(x.shape[1] == 3)


if __name__ == '__main__':
    unittest.main()