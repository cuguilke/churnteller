import os
import json
import warnings

def get_parameters(model, features, use_best=True, path="parameters.json"):
    parameter_dict = None

    if use_best:
        assert os.path.exists(path), "Please run the program again with the keyword '--grid_search'"

        # Load stored (best performing) parameters for the selected model:features duo
        parameter_cache = {}
        if os.path.isfile(path):
            with open(path, "r") as param_file:
                parameter_cache = json.load(param_file)

        model_key = "%s:%s" % (model, features)
        assert model_key in parameter_cache, "Parameter config is not found for %s" % model_key

        parameter_dict = parameter_cache[model_key]

    else:
        if model == "xgboost":
            parameter_dict = {
                "max_depth": range(2, 10, 3),
                "n_estimators": range(60, 220, 40),
                "learning_rate": [0.1, 0.01, 0.001]
            }

        elif model == "svm":
            parameter_dict = {
                "C": [0.1, 1, 10],
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                "kernel": ["linear", "rbf"]
            }

    return parameter_dict

def save_parameters(model, features, parameter_dict, path="parameters.json"):
    model_name = "%s:%s" % (model, features)

    # Load the previous records if exist
    hist_cache = {}
    if os.path.isfile(path):
        with open(path, "r") as hist_file:
            hist_cache = json.load(hist_file)

    # Record new results
    if model_name in hist_cache:
        warnings.warn("%s best parameters are overwritten" % model_name, RuntimeWarning)

    hist_cache[model_name] = parameter_dict

    # Save the updated records
    with open(path, "w+") as hist_file:
        json.dump(hist_cache, hist_file)