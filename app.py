from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
import time

from config import model_path, encoder_path


app = FastAPI()
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

    start_time = time.time()

    try:
        input_data = pd.DataFrame([data.model_dump()])

        input_data["Gender"] = encoders["Gender"].transform(
            input_data["Gender"]
        )

        input_data["Geography"] = encoders["Geography"].transform(
            input_data["Geography"]
        )

        prediction = model.predict(input_data)

        response_time = time.time() - start_time

        logger.info(
            f"Prediction={int(prediction[0])}, "
            f"ResponseTime={response_time:.4f}s"
        )

        return {"prediction": int(prediction[0])}

    except Exception as e:

        logger.error(f"Prediction failed: {str(e)}")

        raise