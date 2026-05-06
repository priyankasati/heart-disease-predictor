import pandas as pd

def load_data():
    df = pd.read_csv("data/final_cardio.csv")
    return df

def split_data(df):
    X = df.drop("cardio", axis=1)
    y = df["cardio"]
    return X, y