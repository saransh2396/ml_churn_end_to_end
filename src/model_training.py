from sklearn.model_selection import train_test_split
from config import dependent_col, test_size, random_state
from src.data_preprocessing import preprocess_data
from src.data_loader import load_data
from sklearn.ensemble import RandomForestClassifier


def train_model(data):
    """
    Train a machine learning model on the preprocessed dataset.

    Args:
        data (pd.DataFrame): The preprocessed dataset.

    Returns:
        tuple: The trained model and the test features and labels.
    """
    # Split the data into features and target variable
    X = data.drop(columns=[dependent_col])
    y = data[dependent_col]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("Model Trained Succesfully")

    return model, X_test, y_test

