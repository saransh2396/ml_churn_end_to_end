from config import drop_cols
import pandas as pd
import numpy as np
from src.data_loader import load_data
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data):
    """
    Preprocess the dataset by dropping specified columns and handling missing values.

    Args:
        data (pd.DataFrame): The input dataset

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Drop specified columns
    data = data.drop(columns=drop_cols, errors='ignore')

    # Handle missing values (if any)
    data = data.fillna(data.mean(numeric_only=True))
    gender_encoder = LabelEncoder()
    geography_encoder = LabelEncoder()

    data['Gender'] = gender_encoder.fit_transform(data['Gender'])
    data['Geography'] = geography_encoder.fit_transform(data['Geography'])

    encoders = {
        "Gender": gender_encoder,
        "Geography": geography_encoder
    }

    return data, encoders


