import logging
from rich.logging import RichHandler
import json
import os

# working_directory = os.path.dirname(os.path.abspath(__file__))
# config_file = working_directory + '/../config.json'
# with open(config_file) as f:
#     config = json.load(f)
# log_file = config['logs']['log_file']

level = logging.DEBUG

logger = logging.getLogger(__name__)

shell_handler = RichHandler()
# file_handler = logging.FileHandler(log_file)

logger.setLevel(level)
shell_handler.setLevel(level)
# file_handler.setLevel(level)

fmt_shell = '%(message)s'
fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

shell_handler.setFormatter(shell_formatter)
# file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
# logger.addHandler(file_handler)
