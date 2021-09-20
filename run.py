import xgboost
import argparse
from models import *
from preprocessing import *
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # Dynamic parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="xgboost", help="supported models: (1) xgboost")
    parser.add_argument("--features", default="RFM", help="supported features for churn prediction: (1) RFM")
    parser.add_argument("--order_data_path", default="./data/machine_learning_challenge_order_data.csv.gz")
    parser.add_argument("--label_data_path", default="./data/machine_learning_challenge_labeled_data.csv.gz")
    parser.add_argument("--grid_search", action="store_true", help="use this command to run grid search, else the best config will be used for testing")

    # Config
    args = vars(parser.parse_args())
    model = args["model"]
    features = args["features"]
    order_data_path = args["order_data_path"]
    label_data_path = args["label_data_path"]
    do_grid_search = args["grid_search"]

    # Load customer data
    customer_info = load_data(order_data_path, label_data_path)

    # Feature extraction
    if features == "RFM":
        x, y = get_RFM_data(customer_info)

    else:
        raise ValueError("%s is not a supported feature" % features)

    # Prepare XGBoost classifier
    if model == "xgboost":
        estimator = xgboost.XGBClassifier(objective="binary:logistic", seed=13, use_label_encoder=False, eval_metric="logloss")
        parameters = get_parameters(model, use_best=not do_grid_search)

    else:
        raise ValueError("%s is not a supported model" % model)

    if do_grid_search:
        # Grid search for the hyperparameters
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=parameters,
                                   scoring="roc_auc",
                                   cv=10,
                                   verbose=True)

        # Start hyperparameter search
        grid_search.fit(x, y)

        print(grid_search.best_estimator_)

    else:
        pass