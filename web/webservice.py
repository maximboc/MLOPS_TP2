from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np
import time
import random
from mlflow.exceptions import RestException

MLFLOW_TRACKING_URI = "http://mlflow-service:8080"
MODEL_NAME = "iris_model"
CURRENT_MODEL_VERSION = 1

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def wait_for_model(model_name: str, version: int, timeout: int = 120):
    start = time.time()
    while True:
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Model {model_name} v{version} loaded successfully!")
            return model
        except RestException:
            print(f"Waiting for model {model_name} v{version} to be registered...")
            time.sleep(3)
            if time.time() - start > timeout:
                raise TimeoutError(f"Model {model_name} v{version} not found after {timeout}s")

current_model = wait_for_model(MODEL_NAME, CURRENT_MODEL_VERSION)
next_model = current_model

p = 0.8

app = FastAPI(title="MLFlow Model Service")

class PredictRequest(BaseModel):
    data: List[List[float]]

@app.post("/predict")
def predict(request: PredictRequest):
    global current_model, next_model, p
    try:
        input_data = np.array(request.data)
        if random.random() < p:
            prediction = current_model.predict(input_data)
            model_used = "current"
        else:
            prediction = next_model.predict(input_data)
            model_used = "next"
        return {"prediction": prediction.tolist(), "model_used": model_used}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/update-model")
def update_model(version: int = CURRENT_MODEL_VERSION):
    global next_model
    try:
        next_model = wait_for_model(MODEL_NAME, version)
        return {"message": f"Next model updated to version {version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model
    try:
        current_model = next_model
        return {"message": "Next model accepted as current"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Accept failed: {e}")
