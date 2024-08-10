from heart_disease_prediction.components.estimator import Estimator
from heart_disease_prediction.exception import heart_disease_prediction_exception
import pandas as pd
from heart_disease_prediction.constants import BEST_MODEL_PATH, PREPROCESSOR_PATH


class PredictionPipeline:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_Data(self):
        if self.args:

            if len(self.args) == 1 and isinstance(self.args[0], pd.DataFrame):
                print("Input is a dataframe")
                return self.args[0]
            else:
                raise heart_disease_prediction_exception("Invalid Arguments")

        elif self.kwargs:
            return pd.DataFrame.from_dict(self.kwargs)

        else:
            raise heart_disease_prediction_exception("Invalid Arguments")

    def predict(self):
        data = self.get_Data()
        out = Estimator(
            testData=data,
            best_model_path=BEST_MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
        ).run()
        return out
