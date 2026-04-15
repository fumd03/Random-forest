import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "penguins_size.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

TEST_SIZE = 0.2
RANDOM_STATE = 42
