from heart_disease_prediction.pipline.training_pipeline import TrainPipeline
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from heart_disease_prediction.constants import BEST_MODEL_PATH, PREPROCESSOR_PATH
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

test_df = pd.read_csv("/home/amadgakkhar/code/MLOps-Project/sample_test.csv")


# TrainPipeline().start_training()

# estimate = Estimator(
#     testData=test_df,
#     preprocessor_path=PREPROCESSOR_PATH,
#     best_model_path=BEST_MODEL_PATH,
# ).run()

estimate = PredictionPipeline(1).predict()
print(estimate)
