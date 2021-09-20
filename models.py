import os

def get_parameters(model_name, use_best=True):
    parameter_dict = None

    if use_best:
        assert os.path.exists("params.config"), "Please run the program again with the keyword '--grid_search'"

    else:
        if model_name == "xgboost":
            parameter_dict = {
                "max_depth": range(2, 10, 1),
                "n_estimators": range(60, 220, 40),
                "learning_rate": [0.1, 0.01, 0.001, 0.0001]
            }

    return parameter_dict