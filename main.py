from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from sklearn.metrics import accuracy_score
import pickle
from config import model_path,encoder_path


data = load_data()
preprocessed_data, encoders = preprocess_data(data)
model, X_test, y_test = train_model(preprocessed_data)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

with open(model_path, "wb") as model_file:
	pickle.dump(model, model_file)
      
with open("models/encoders.pkl", "wb") as encoder_file:
    pickle.dump(encoders, encoder_file)

print(f"Model saved to {model_path}")



