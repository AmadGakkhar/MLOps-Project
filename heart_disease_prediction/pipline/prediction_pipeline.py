from heart_disease_prediction.components.estimator import Estimator
from heart_disease_prediction.exception import heart_disease_prediction_exception
import pandas as pd
import json
from heart_disease_prediction.constants import BEST_MODEL_PATH, PREPROCESSOR_PATH
from heart_disease_prediction.utils.main_utils import df_to_json, json_to_df


class PredictionPipeline:
    def __init__(self, data_json):

        self.data_json = data_json

    def get_Data(self):
        try:
            json.loads(self.data_json)  ## Checks if the data is in json format
            df = json_to_df(self.data_json)  ## Converts
            # print("Converted Json to DF")

            return df
        except:

            if isinstance(self.data_json, pd.DataFrame):
                df = self.data_json
                # print("Data already in DF")
                return df
            else:
                raise Exception("Invalid Data Format in the Prediction Pipeline")

    def predict(self):
        data = self.get_Data()
        # print(data)
        out = Estimator(
            testData=data,
            best_model_path=BEST_MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
        ).run()
        arr_list = out.tolist()
        json_data = json.dumps(arr_list)

        return json_data
