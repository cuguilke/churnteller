import xgboost
import argparse
from tools import *
from models import *
from preprocessing import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
    # Dynamic parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="xgboost", help="supported models: (1) xgboost, (2) svm")
    parser.add_argument("--features", default="RFM", help="supported features for churn prediction: (1) RFM, (2) custom")
    parser.add_argument("--order_data_path", default="./data/machine_learning_challenge_order_data.csv.gz")
    parser.add_argument("--label_data_path", default="./data/machine_learning_challenge_labeled_data.csv.gz")
    parser.add_argument("--parameter_path", default="parameters.json")
    parser.add_argument("--grid_search", action="store_true", help="use this command to run grid search, else the best config will be used for testing")
    parser.add_argument("--print_config", action="store_true")

    # Config
    args = vars(parser.parse_args())
    model = args["model"]
    features = args["features"]
    order_data_path = args["order_data_path"]
    label_data_path = args["label_data_path"]
    parameter_path = args["parameter_path"]
    do_grid_search = args["grid_search"]
    print_config = args["print_config"]

    # Print out the active configuration
    if print_config:
        log_config(args)

    # Load customer data
    customer_info = load_data(order_data_path, label_data_path)
    log("Customer info is loaded...")

    # Feature extraction
    if features == "RFM":
        x, y, x_test, y_test = get_RFM_data(customer_info, final_test=not do_grid_search)

    elif features == "custom":
        x, y, x_test, y_test = get_custom_data(customer_info, final_test=not do_grid_search)

    else:
        raise ValueError("%s is not a supported feature" % features)
    log("Customer data is prepared...")

    # Prepare the classifier
    if model == "xgboost":
        estimator = xgboost.XGBClassifier(objective="binary:logistic", seed=13, use_label_encoder=False, eval_metric="logloss")

    elif model == "svm":
        estimator = SVC()

    else:
        raise ValueError("%s is not a supported model" % model)
    log("Model is initialized...")

    # Get parameters (best | grid search)
    parameters = get_parameters(model, features, use_best=not do_grid_search, path=parameter_path)

    if do_grid_search:
        # Grid search for the hyperparameters
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=parameters,
                                   scoring="roc_auc",
                                   n_jobs=2,
                                   cv=10,
                                   verbose=True)
        log("Grid search is initialized...")

        # Start hyperparameter search
        grid_search.fit(x, y)
        log("Grid search is completed...")

        # Save the best params
        save_parameters(model, features, grid_search.best_params_, path=parameter_path)
        log("The best performing parameters are saved.")

    else:
        # Set the recorded best parameters
        estimator.set_params(**parameters)
        log("The parameter storage is used to load the best performing setting...")

        # Training
        log("Training starts...")
        estimator.fit(x, y)
        log("Model training is completed.")

        # Testing
        y_pred = estimator.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1_score = f1_score(y_test, y_pred)
        log("%s:%s test acc: %.2f, recall: %.2f, precision: %.2f, f1 score: %.2f" % (model, features, acc, recall, precision, f1_score))
        log("Customer churn rate: %.2f" % get_customer_churn_rate(y_test))

        # Record experiment results
        record_results(model, features, {"acc": acc, "recall": recall, "precision": precision, "f1_score": f1_score})
        log("Empirical results are recorded.")
