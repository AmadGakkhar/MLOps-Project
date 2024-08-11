from heart_disease_prediction.pipline.training_pipeline import TrainPipeline
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

load_dotenv()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
def train_pipeline(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.get("/train/start")
def start_training(request: Request):
    TrainPipeline().start_training()
    return templates.TemplateResponse("train_complete.html", {"request": request})


@app.get("/test")
def test_pipeline_get(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})


@app.post("/test")
async def test_pipeline(request: Request):
    data = await request.json()
    data_df = pd.DataFrame([data])
    print(data_df)
    estimate = PredictionPipeline(data_df).predict()
    print(estimate)

    return estimate


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
