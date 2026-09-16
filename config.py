path = "data/Churn_Modelling.csv"
drop_cols = ['RowNumber','CustomerId','Surname']
dependent_col = 'Exited'
test_size = 0.2
random_state = 42
model_path = "models/model.pkl"
encoder_path = "models/encoders.pkl"
