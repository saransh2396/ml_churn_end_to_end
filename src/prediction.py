import pickle
from config import model_path


def load_model():
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def predict(input_data):
    model = load_model()

    prediction = model.predict(input_data)

    return prediction