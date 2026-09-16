from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

from config import model_path, encoder_path


app = FastAPI()


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
    return {"message": "Churn prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.model_dump()])

    # Apply saved encoders
    input_data["Gender"] = encoders["Gender"].transform(
        input_data["Gender"]
    )

    input_data["Geography"] = encoders["Geography"].transform(
        input_data["Geography"]
    )

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }