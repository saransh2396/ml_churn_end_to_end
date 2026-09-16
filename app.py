from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
import time

from config import model_path, encoder_path


app = FastAPI()

total_requests = 0
successful_requests = 0
failed_requests = 0
response_times = []

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# Load model
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)


# Load encoders
with open(encoder_path, "rb") as encoder_file:
    encoders = pickle.load(encoder_file)


class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    return {"message": "Churn prediction API is running - CI/CD test"}


@app.post("/predict")
def predict(data: CustomerData):
    global total_requests, successful_requests, failed_requests, response_times

    start_time = time.perf_counter()
    total_requests += 1

    try:
        input_data = pd.DataFrame([data.model_dump()])

        input_data["Gender"] = encoders["Gender"].transform(
            input_data["Gender"]
        )

        input_data["Geography"] = encoders["Geography"].transform(
            input_data["Geography"]
        )

        prediction = model.predict(input_data)

        response_time = time.perf_counter() - start_time
        response_times.append(response_time)

        successful_requests += 1

        logger.info(
            f"Prediction={int(prediction[0])}, "
            f"ResponseTime={response_time:.4f}s"
        )

        return {"prediction": int(prediction[0])}

    except Exception as e:
        failed_requests += 1

        logger.error(f"Prediction failed: {str(e)}")

        raise

@app.get("/metrics")
def metrics():
    if response_times:
        average_latency = sum(response_times) / len(response_times)
    else:
        average_latency = 0

    error_rate = (
        failed_requests / total_requests * 100
        if total_requests > 0
        else 0
    )

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate_percent": round(error_rate, 2),
        "average_latency_seconds": round(average_latency, 4)
    }