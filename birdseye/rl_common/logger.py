"""
These functions are adapted from github.com/Officium/RL-Experiments

"""
"""
Logger singleton wrapper
Default logger folder is `os.path.join(__file__, '..', '..', 'logs')`
"""
import logging
import logging.handlers
import os


__all__ = ['init_logger']


def init_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_level = logging.INFO
    log_format = '%(levelname)s %(asctime)s %(processName)s %(process)s' \
                 '| %(filename)s | [line:%(lineno)d] | %(message)s'

    logger = logging.getLogger(log_dir)
    logger.setLevel(log_level)
    path = os.path.join(log_dir, 'main.log')

    # file handler (log file)
    log_handler = logging.handlers.RotatingFileHandler(filename=path)
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    # stream handler (default sys.stderr)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    return logger

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

