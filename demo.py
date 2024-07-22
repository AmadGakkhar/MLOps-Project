from heart_disease_prediction.logger import logging
from heart_disease_prediction.exception import heart_disease_prediction_exception
import sys

# logging.info("Hello from the Logs")

try:
    a = 2 / 0
except Exception as e:
    raise heart_disease_prediction_exception(e, sys)
