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
from heart_disease_prediction.constants import BEST_MODEL_PATH

# from heart_disease_prediction.entity.estimator import USvisaModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _get_model_list(self, model_schema):
        model_names = list(model_schema["model_selection"].keys())
        # print(model_names)
        model_list = []
        params_list = []
        for model in model_names:
            module_name = model_schema["model_selection"][model]["module"]
            class_name = model_schema["model_selection"][model]["class"]
            module = importlib.import_module(module_name)

            class_ = getattr(module, class_name)
            classifier = class_()
            model_list.append(classifier)
            params = model_schema["model_selection"][model]["search_param_grid"]
            params_list.append((params))

        return model_list, params_list

    def _evaluate_models(self, X_train, y_train, X_test, y_test, model_config_path):

        model_schema = read_yaml_file(file_path=model_config_path)
        # print(type(model_schema))
        models_list, params_list = self._get_model_list(model_schema)
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        auc = []
        models_tuned = {}
        tuned_models_list = []
        for model, params in zip(models_list, params_list):
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                verbose=3,
            )
            grid.fit(X_train, y_train)
            models_tuned[model] = grid.best_params_

        for model in list(models_tuned.keys()):

            tuned_model = model.set_params(**models_tuned[model])
            tuned_model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = tuned_model.predict(X_train)
            y_test_pred = tuned_model.predict(X_test)

            # Training set performance
            (
                model_train_accuracy,
                model_train_f1,
                model_train_precision,
                model_train_recall,
                model_train_rocauc_score,
            ) = evaluate_clf(y_train, y_train_pred)

            # Test set performance
            (
                model_test_accuracy,
                model_test_f1,
                model_test_precision,
                model_test_recall,
                model_test_rocauc_score,
            ) = evaluate_clf(y_test, y_test_pred)

            print(tuned_model)
            tuned_models_list.append(tuned_model)

            print("Model performance for Training set")
            print("- Accuracy: {:.4f}".format(model_train_accuracy))
            print("- F1 score: {:.4f}".format(model_train_f1))
            print("- Precision: {:.4f}".format(model_train_precision))
            print("- Recall: {:.4f}".format(model_train_recall))
            print("- Roc Auc Score: {:.4f}".format(model_train_rocauc_score))

            print("----------------------------------")

            print("Model performance for Test set")
            print("- Accuracy: {:.4f}".format(model_test_accuracy))
            accuracy_list.append(model_test_accuracy)
            print("- F1 score: {:.4f}".format(model_test_f1))
            print("- Precision: {:.4f}".format(model_test_precision))
            print("- Recall: {:.4f}".format(model_test_recall))
            print("- Roc Auc Score: {:.4f}".format(model_test_rocauc_score))
            auc.append(model_test_rocauc_score)
            f1_list.append(model_test_f1)
            precision_list.append(model_test_precision)
            recall_list.append(model_test_recall)
            print("=" * 35)
            print("\n")

        report = (
            pd.DataFrame(
                list(
                    zip(
                        tuned_models_list,
                        accuracy_list,
                        params_list,
                        f1_list,
                        precision_list,
                        recall_list,
                    )
                ),
                columns=["Model", "Accuracy", "Params", "F1", "Precision", "Recall"],
            )
            .sort_values(by=["Accuracy"], ascending=False)
            .reset_index(drop=True)
        )
        final_model = report["Model"][0]
        final_model_accuracy = report["Accuracy"][0]
        final_model_precision = report["Precision"][0]
        final_model_f1 = report["F1"][0]
        final_model_recall = report["Recall"][0]
        classification_artifact = ClassificationMetricArtifact(
            accuracy=final_model_accuracy,
            f1_score=final_model_f1,
            precision_score=final_model_precision,
            recall_score=final_model_recall,
        )
        return report, final_model, final_model_accuracy, classification_artifact

    def get_model_object_and_report(self, train: np.array, test: np.array):
        """
        Method Name :   get_model_object_and_report
        Description :   This function gets the best model object and report of the best model

        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # logging.info("Using neuro_mf to get best model object and report")
            # model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1],
            )
            report, final_model, accuracy, classification_artifact = (
                self._evaluate_models(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    self.model_trainer_config.model_config_file_path,
                )
            )

            return report, final_model, accuracy, classification_artifact

        except Exception as e:
            raise heart_disease_prediction_exception(e, sys) from e

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            # best_model_detail, metric_artifact = self.get_model_object_and_report(
            #     train=train_arr, test=test_arr
            # )
            report, model, accuracy, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )
            save_object(self.model_trainer_config.trained_model_file_path, model)

            if accuracy > self.model_trainer_config.expected_accuracy:
                save_object(BEST_MODEL_PATH, model)
                logging.info("Best model found with score more than base score")
                print("Best model found with score more than base score")
            else:
                print("No model found with score more than base score")
                logging.info("No model found with score more than base score")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            return model, model_trainer_artifact

        #     logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        #     return model_trainer_artifact
        except Exception as e:
            raise heart_disease_prediction_exception(e, sys) from e
