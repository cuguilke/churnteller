import os
import pandas
import numpy as np

def load_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load and prepare the training data
    data = pandas.read_csv(os.path.join(ROOT_DIR, "./data/machine_learning_challenge_order_data.csv.gz"))

    customer_info = {}
    for i, entry in data.iterrows():
        temp = {i: entry[i] for i in entry.keys() if i != "customer_id"}
        if entry["customer_id"] not in customer_info:
            customer_info[entry["customer_id"]] = {"label": None, "data": []}
        customer_info[entry["customer_id"]]["data"].append(temp)

    # Load and prepare test data
    test_data = pandas.read_csv(os.path.join(ROOT_DIR, "./data/machine_learning_challenge_labeled_data.csv.gz"))

    for i, entry in test_data.iterrows():
        customer_id = entry["customer_id"]
        label = entry["is_returning_customer"]

        assert customer_id in customer_info, "Missing customer in order data!"
        customer_info[customer_id]["label"] = label

    return customer_info