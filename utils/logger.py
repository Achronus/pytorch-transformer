import logging
import os

from logging import FileHandler, Formatter, StreamHandler


def create_logger(name: str, filename: str, flag: bool, mode: str = 'w') -> logging.Logger:
    """
    Creates and returns a basic logger with a predefined file handler.

    :param name: (str) name of the logger
    :param filename: (str) log document filename. Automatically prepended `.log`. Stored in `logs/`
    :param flag: (bool) flag for enabling or disabling the logger
    :param mode: (optional, str) sets the file handler mode. Defaults to overwriting the file content
    """
    filepath = f'{os.getcwd()}/logs/{filename}.log'

    # Create log file if it doesn't exist
    if not os.path.isfile(filepath):
        with open(filepath, 'w') as f:
            pass

    # Create generic logger and return it
    log_obj = Logger(name, enable=flag)
    log_obj.add_fh(filepath, mode=mode)
    return log_obj.get()


class Logger:
    """
    A class dedicated to creating custom loggers.

    :param name: (str) name of the logger
    :param level: (optional, int | str) the desired logger level. Effects console handler. Defaults to INFO
    :param enable: (optional, bool) a flag for enabling or disabling the logger. Defaults to False
    """
    def __init__(self, name: str, level: int | str = logging.INFO, enable: bool = False) -> None:
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.disabled = not enable

        # Set other variables
        self.level = level
        self.formatter = Formatter("%(name)s:%(levelname)s:%(message)s")
        self.date_formatter = Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S")

    def get(self) -> logging.Logger:
        """Retrieves the logger."""
        return self.logger

    def add_fh(self, filename: str, mode: str = 'a', level: int | str = None, formatter: Formatter = None) -> None:
        """
        Adds a file handler to the logger.

        :param filename: (str) the filename to log to
        :param mode: (optional, str) select the mode for accessing the log file. Defaults to 'a' (append)
        :param level: (optional, int | str) the desired logging level. Defaults to logger level
        :param formatter: (optional, logging.Formatter) the format of the file handler. Defaults to logger format
        """
        fh = FileHandler(filename, mode=mode)
        fh.setLevel(self.level) if level is None else fh.setLevel(level)
        fh.setFormatter(self.formatter) if formatter is None else fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def add_sh(self, level: int | str = None, formatter: Formatter = None) -> None:
        """
        Adds a stream (console) handler to the logger.

        :param level: (optional, int | str) the desired logging level. Defaults to logger level
        :param formatter: (optional, logging.Formatter) the format of the file handler. Defaults to logger format
        """
        sh = StreamHandler()
        sh.setLevel(self.level) if level is None else sh.setLevel(level)
        sh.setFormatter(self.formatter) if formatter is None else sh.setFormatter(formatter)
        self.logger.addHandler(sh)
