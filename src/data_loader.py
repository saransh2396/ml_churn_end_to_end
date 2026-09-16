import pandas as pd
from config import path

def load_data():
    """
    Load the dataset from the specified path.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"File not found at {path}. Please check the path and try again.")
        return None


