import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/final_cardio.csv")

X = df.drop("cardio", axis=1)
y = df["cardio"]

model = joblib.load("models/xgb_model.pkl")

y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))