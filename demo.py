import pandas as pd
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from heart_disease_prediction.pipline.training_pipeline import TrainPipeline
from dotenv import load_dotenv

load_dotenv()

test_df = pd.read_csv("/home/amadgakkhar/code/MLOps-Project/sample_test.csv")
print("Original DF \n\n")
print(test_df)

TrainPipeline().start_training()
estimate = PredictionPipeline(test_df).predict()


print(estimate)
