from heart_disease_prediction.pipline.training_pipeline import TrainPipeline
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from heart_disease_prediction.constants import BEST_MODEL_PATH, PREPROCESSOR_PATH
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

load_dotenv()

test_df = pd.read_csv("/home/amadgakkhar/code/MLOps-Project/sample_test.csv")


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the app!"}


@app.get("/train")
def train_pipeline():
    TrainPipeline().start_training()
    return {"message": "Training Completed"}


# @app.post("/test")
# def test_pipeline(test_data: pd.DataFrame):
#     estimate = PredictionPipeline(test_data).predict()
#     print(estimate)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# TrainPipeline().start_training()

# estimate = Estimator(
#     testData=test_df,
#     preprocessor_path=PREPROCESSOR_PATH,
#     best_model_path=BEST_MODEL_PATH,
# ).run()

# estimate = PredictionPipeline(1).predict()
# print(estimate)
