import sys
from typing import Tuple
import importlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from heart_disease_prediction.exception import heart_disease_prediction_exception
from heart_disease_prediction.logger import logging
from heart_disease_prediction.utils.main_utils import (
    load_numpy_array_data,
    read_yaml_file,
    load_object,
    save_object,
    evaluate_clf,
)
from heart_disease_prediction.entity.config_entity import ModelTrainerConfig
from heart_disease_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)


class Estimator:
    def __init__(
        self,
        testData: DataFrame,
        preprocessor_path: str,
        best_model_path: str,
    ):
        """
        :testData: Data to get the prediction for
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param model_trainer_artifact: Output reference of model trainer artifact stage
        """
        self.preprocessor_path = preprocessor_path
        self.best_model_path = best_model_path
        self.data = testData

    def estimate(self, data: DataFrame):
        data.dropna(axis=0, inplace=True)

        preprocessor = load_object(
            self.preprocessor_path
        )
        model = load_object(self.best_model_path)

        # print(
        #     "Estimator: Shape of data after transfomration",
        #     np.shape(data),
        # )

        input_feature_arr = preprocessor.transform(data)
        # print(
        #     "Estimator: Shape of data after transfomration",
        #     np.shape(input_feature_arr),
        # )
        return model.predict(input_feature_arr)

    def run(self):
        out = self.estimate(self.data)
        return out
