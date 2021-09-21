import logging
from enum import Enum
from time import localtime, strftime

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