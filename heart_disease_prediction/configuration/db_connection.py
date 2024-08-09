import sys

from heart_disease_prediction.exception import heart_disease_prediction_exception
from heart_disease_prediction.logger import logging

import os
from heart_disease_prediction.constants import DATABASE_NAME
import pymongo
import certifi

ca = certifi.where()


class MongoDBClient:

    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                MONGODB_URL_KEY = os.getenv("MONGODB_URL")

                if MONGODB_URL_KEY is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(
                    MONGODB_URL_KEY, tlsCAFile=ca
                )
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise heart_disease_prediction_exception(e, sys)
