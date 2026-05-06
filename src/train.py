import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("data/final_cardio.csv")

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgb_model.pkl")

print("Model saved successfully!")