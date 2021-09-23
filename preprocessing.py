import os
import pandas
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedShuffleSplit

DATE_FORMAT = "%Y-%m-%d"
MIN_DATE = datetime.strptime("2011-05-11", DATE_FORMAT)
LAST_DATE = datetime.strptime("2017-02-28", DATE_FORMAT)

def load_data(order_data_path, label_data_path):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load and prepare the training data
    data = pandas.read_csv(os.path.join(ROOT_DIR, order_data_path))

    customer_info = {}
    for i, entry in data.iterrows():
        temp = {i: entry[i] for i in entry.keys() if i != "customer_id"}
        if entry["customer_id"] not in customer_info:
            customer_info[entry["customer_id"]] = {"label": None, "data": []}
        customer_info[entry["customer_id"]]["data"].append(temp)

    # Load and prepare test data
    test_data = pandas.read_csv(os.path.join(ROOT_DIR, label_data_path))

    for i, entry in test_data.iterrows():
        customer_id = entry["customer_id"]
        label = entry["is_returning_customer"]

        assert customer_id in customer_info, "Missing customer in order data!"
        customer_info[customer_id]["label"] = 0 if label == 1 else 1

    return customer_info

def get_train_val_split(x, y, val_split=0.1, n_splits=10):
    assert val_split <= 1, "Validation split rate cannot be greater than 1"

    split_list = []
    train_val_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_split, random_state=13)

    # Balanced train/val split so that returning customers are evenly (almost) distributed between splits
    for i_train, i_val in train_val_split.split(x, y):
        x_train, x_val = x[i_train], x[i_val]
        y_train, y_val = y[i_train], y[i_val]
        split_list.append(((x_train, y_train), (x_val, y_val)))

    return split_list

def get_customer_churn_rate(y):
    return np.sum(y) / y.shape[0]

def get_RFM_data(customer_info, final_test=False):
    """
    Recency, Frequency, Monetary model data preparation

    #Arguments
        :param customer_info: (dict) imported CSV data
        :param final_test: (bool) if False, generate custom labels using the last 6 months

    """
    x = []
    y = []

    # Use the last 6 months for cross-validation
    threshold_date = LAST_DATE - timedelta(days=180)

    for customer in customer_info:
        temp = []
        last_order_date = MIN_DATE
        for entry in customer_info[customer]["data"]:
            new_label = 1

            order_date = datetime.strptime(entry["order_date"], DATE_FORMAT)
            if order_date <= threshold_date:

                # Ignore failed orders
                if entry["is_failed"] == 0:
                    temp.append(entry["amount_paid"])
                    last_order_date = max(last_order_date, order_date)

            else:
                new_label = 0

        # Eliminate customers that made their first order in the last 6 months
        if last_order_date > MIN_DATE:
            x.append([(threshold_date - last_order_date).days, sum(temp), len(temp)])
            y.append(new_label)

    # List to numpy array
    x = np.array(x).astype("float32")
    y = np.array(y)

    x_test = []
    y_test = []
    if final_test:
        for customer in customer_info:
            temp = []
            last_order_date = MIN_DATE
            for entry in customer_info[customer]["data"]:
                order_date = datetime.strptime(entry["order_date"], DATE_FORMAT)

                # Ignore failed orders
                if entry["is_failed"] == 0:
                    temp.append(entry["amount_paid"])
                    last_order_date = max(last_order_date, order_date)

            x_test.append([(LAST_DATE - last_order_date).days, sum(temp), len(temp)])
            y_test.append(customer_info[customer]["label"])

        x_test = np.array(x_test).astype("float32")
        y_test = np.array(y_test)

    return x, y, x_test, y_test

def get_custom_data(customer_info, final_test=False):
    """
    Custom model data preparation

    #Arguments
        :param customer_info: (dict) imported CSV data
        :param final_test: (bool) if False, generate custom labels using the last 6 months

    """
    x = []
    y = []

    # Unique IDs for misc info (ID to index mapping)
    payment_IDs = {1779: 0, 1619: 1, 1491: 2, 1811: 3, 1523: 4}
    platform_IDs = {30231: 0, 30359: 1, 29463: 2, 29815: 3, 30423: 4, 30391: 5, 29495: 6, 525: 7, 29751: 8, 30199: 9, 30135: 10, 22263: 11, 22167: 12, 22295: 13}
    transmission_IDs = {4356: 0, 4324: 1, 4228: 2, 4260: 3, 4196: 4, 212: 5, 4996: 6, 21124: 7, 1988: 8, 2020: 9}

    # Use the last 6 months for cross-validation
    threshold_date = LAST_DATE - timedelta(days=180)

    for customer in customer_info:
        temp = []
        total_voucher = 0
        payment_freq = [0] * 5
        platform_freq = [0] * 14
        tranmission_freq = [0] * 10

        last_order_date = MIN_DATE
        for entry in customer_info[customer]["data"]:
            new_label = 1

            order_date = datetime.strptime(entry["order_date"], DATE_FORMAT)
            if order_date <= threshold_date:

                # Ignore failed orders
                if entry["is_failed"] == 0:
                    temp.append(entry["amount_paid"])
                    total_voucher += entry["voucher_amount"]
                    payment_freq[payment_IDs[entry["payment_id"]]] += 1
                    platform_freq[platform_IDs[entry["platform_id"]]] += 1
                    tranmission_freq[transmission_IDs[entry["transmission_id"]]] += 1
                    last_order_date = max(last_order_date, order_date)

            else:
                new_label = 0

        # Eliminate customers that made their first order in the last 6 months
        if last_order_date > MIN_DATE:
            feature_vector = [(threshold_date - last_order_date).days, sum(temp), len(temp), total_voucher]
            feature_vector.extend(payment_freq)
            feature_vector.extend(platform_freq)
            feature_vector.extend(tranmission_freq)
            x.append(feature_vector)
            y.append(new_label)

    # List to numpy array
    x = np.array(x).astype("float32")
    y = np.array(y)

    x_test = []
    y_test = []
    if final_test:
        for customer in customer_info:
            temp = []
            total_voucher = 0
            payment_freq = [0] * 5
            platform_freq = [0] * 14
            tranmission_freq = [0] * 10

            last_order_date = MIN_DATE
            for entry in customer_info[customer]["data"]:
                order_date = datetime.strptime(entry["order_date"], DATE_FORMAT)

                # Ignore failed orders
                if entry["is_failed"] == 0:
                    temp.append(entry["amount_paid"])
                    total_voucher += entry["voucher_amount"]
                    payment_freq[payment_IDs[entry["payment_id"]]] += 1
                    platform_freq[platform_IDs[entry["platform_id"]]] += 1
                    tranmission_freq[transmission_IDs[entry["transmission_id"]]] += 1
                    last_order_date = max(last_order_date, order_date)

            # Eliminate customers that only have failed orders
            if last_order_date > MIN_DATE:
                feature_vector = [(LAST_DATE - last_order_date).days, sum(temp), len(temp), total_voucher]
                feature_vector.extend(payment_freq)
                feature_vector.extend(platform_freq)
                feature_vector.extend(tranmission_freq)
                x_test.append(feature_vector)
                y_test.append(customer_info[customer]["label"])

        x_test = np.array(x_test).astype("float32")
        y_test = np.array(y_test)

    return x, y, x_test, y_test



