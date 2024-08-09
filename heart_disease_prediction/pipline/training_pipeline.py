from heart_disease_prediction.logger import logging
from heart_disease_prediction.exception import heart_disease_prediction_exception
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_validation import DataValidation
from heart_disease_prediction.components.data_transformation import DataTransformation
from heart_disease_prediction.components.model_trainer import ModelTrainer
from heart_disease_prediction.components.estimator import Estimator
import pandas as pd


from heart_disease_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

import sys
from dotenv import load_dotenv
import os


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_training(self):

        data_ingestion_artifact = DataIngestion(
            DataIngestionConfig
        ).initiate_data_ingestion()

        data_validation_artifact = DataValidation(
            data_validation_config=DataValidationConfig,
            data_ingestion_artifact=data_ingestion_artifact,
        ).initiate_data_validation()

        data_transformation_artifact = DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=DataTransformationConfig,
        ).initiate_data_transformation()

        model, model_trainer_artifact = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=ModelTrainerConfig,
        ).initiate_model_trainer()

        return model_trainer_artifact, data_transformation_artifact
