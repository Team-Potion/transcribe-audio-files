#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging


def configure_logging(log_level: int, log_file: str = None) -> None:
    """
    Configure the logger with a specified log level and an optional log file. If a log file is provided,
        the logs will be written to the file as well as  to the console.

    Args:
        log_level (int, optional): The desired logging level (e.g., logging.INFO, logging.DEBUG, etc.). 
        log_file (str, optional): The path to the log file where logs will be written. If not provided, 
            logs will be written to the console. Default is None.

    Returns:
        None
    """
    # define the log format, including timestamp, log level, and message
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # create a console handler and set its log level and format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # create a file handler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

    # get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    if log_file:
        root_logger.addHandler(file_handler)


def get_logger(name: str, log_level: int = logging.WARNING, log_file: str = None) -> logging.Logger:
    """
    Get a logger with a specific name.

    Args:
        name (str): The desired name for the logger.
        log_level (int, optional): The desired logging level (e.g., logging.INFO, logging.DEBUG, etc.). 
            Default is logging.INFO.
        log_file (str, optional): The path to the log file where logs will be written. If not provided, 
            logs will be written to the console. Default is None.

    Returns:
        logging.Logger: The logger object.
    """

    # configure logging with the specified log level and log file
    configure_logging(log_level = log_level, log_file = log_file)

    # get the logger with the specified name
    logger = logging.getLogger(name)

    return logger
