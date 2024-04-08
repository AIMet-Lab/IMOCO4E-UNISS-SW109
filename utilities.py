
import logging


def instantiate_logger(filepath: str):

    logger_path = filepath

    stream_logger = logging.getLogger("pynever.strategies.verification")
    file_logger = logging.getLogger("Log File")

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logger_path, 'a+')

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    stream_logger.addHandler(stream_handler)
    file_logger.addHandler(file_handler)

    stream_logger.setLevel(logging.INFO)
    file_logger.setLevel(logging.INFO)

    return stream_logger, file_logger
