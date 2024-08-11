from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from heart_disease_prediction.pipline.training_pipeline import TrainPipeline
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import uvicorn
import pandas as pd

load_dotenv()

app = FastAPI()


class TestData(BaseModel):
    gender: str
    age: float
    hypertension: float
    heart_disease: float
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the app!"}


@app.get("/train")
def train_pipeline():
    TrainPipeline().start_training()
    return {"message": "Training pipeline completed!"}


@app.post("/test")
def test_pipeline(test_data: TestData):
    # Run prediction pipeline here with test_data
    df = pd.DataFrame([test_data.dict()])
    estimate = PredictionPipeline(df).predict()
    prediction_output = estimate
    if prediction_output == "[0.0]":
        prediction_output = "No Risk of Stroke"
    elif prediction_output == "[1.0]":
        prediction_output = "High Risk of Stroke"

    return {
        "message": "Prediction pipeline completed!",
        "prediction": prediction_output,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
