import os
import json
import logging
from enum import Enum
from time import localtime, strftime

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.log"), filemode="a", format="%(message)s", level=logging.INFO)

def get_time():
    return "[%s]" % strftime("%a, %d %b %Y %X", localtime())

class LogType(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

def log(msg, log_type=LogType.INFO, to_file=True, to_stdout=True):
    msg = "%s %s" % (get_time(), msg)

    if to_stdout:
        print(msg)
    if to_file and log_type == LogType.DEBUG:
        logging.debug(msg)
    elif to_file and log_type == LogType.INFO:
        logging.info(msg)
    elif to_file and log_type == LogType.WARNING:
        logging.warning(msg)
    elif to_file and log_type == LogType.ERROR:
        logging.error(msg)

def log_config(config):
    log("Active Configuration:")
    log("--------------------")
    for key in config:
        residual = max(24 - len(key), 0)
        temp = ""
        while len(temp) < residual:
            temp += " "
        log("%s%s: %s" % (key, temp, config[key]))

def record_results(model, features, result_dict, path="results.json"):
    if result_dict is not None:
        model_name = "%s:%s" % (model, features)

        # Load the previous records if exist
        hist_cache = {}
        if os.path.isfile(path):
            with open(path, "r") as hist_file:
                hist_cache = json.load(hist_file)

        # Record new results
        if model_name in hist_cache:
            hist_cache[model_name].append(result_dict)
        else:
            hist_cache[model_name] = [result_dict]

        # Save the updated records
        with open(path, "w+") as hist_file:
            json.dump(hist_cache, hist_file)

def normalize(vector):
    return [float(i) / sum(vector) for i in vector]