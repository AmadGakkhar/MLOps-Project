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
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from heart_disease_prediction.utils.main_utils import df_to_json
import sys
from dotenv import load_dotenv
import os

load_dotenv()

test_df = pd.read_csv("/home/amadgakkhar/code/MLOps-Project/sample_test.csv")
print("Original DF \n\n")
print(test_df)
# test_data = df_to_json(test_df)
test_dict = test_df.to_dict(orient="list")
# print(test_dict)
# print("\n\nDataFrame\n\n")
# print(pd.DataFrame(test_dict))
estimate = PredictionPipeline(test_dict).predict()

# data_ingestion_artifact = DataIngestion(DataIngestionConfig).initiate_data_ingestion()


# data_validation_artifact = DataValidation(
#     data_validation_config=DataValidationConfig,
#     data_ingestion_artifact=data_ingestion_artifact,
# ).initiate_data_validation()


# data_transformation_artifact = DataTransformation(
#     data_ingestion_artifact=data_ingestion_artifact,
#     data_validation_artifact=data_validation_artifact,
#     data_transformation_config=DataTransformationConfig,
# ).initiate_data_transformation()


# model, trainer_artifact = ModelTrainer(
#     data_transformation_artifact=data_transformation_artifact,
#     model_trainer_config=ModelTrainerConfig,
# ).initiate_model_trainer()


# estimate = Estimator(
#     testData=test_df,
#     data_transformation_artifact=data_transformation_artifact,
#     model_trainer_artifact=trainer_artifact,
# ).run()

print(estimate)
# Json Objec
# {
#     "gender": "",
#     "age": "",
#     "hypertension": "",
#     "heart_disease": "",
#     "ever_married": "",
#     "work_type": "",
#     "Residence_type": "",
#     "avg_glucose_level": "",
#     "bmi": "",
#     "smoking_status": ""
# }