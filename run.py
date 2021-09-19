from preprocessing import *

if __name__ == '__main__':
    order_data_path = "./data/machine_learning_challenge_order_data.csv.gz"
    label_data_path = "./data/machine_learning_challenge_labeled_data.csv.gz"

    customer_info = load_data(order_data_path, label_data_path)
    x, y = get_RFM_data(customer_info)