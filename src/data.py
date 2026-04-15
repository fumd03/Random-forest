import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, test_size, random_state):
    path = "../data/penguins_size.csv"
    df = pd.read_csv(path)

    # Replace NA with proper NaN
    df = df.replace("NA", pd.NA)
    df = df.dropna()

    # Features
    X = df.drop("species", axis=1)
    y = df["species"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
