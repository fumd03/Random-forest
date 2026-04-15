import pandas as pd
from src.utils import load_model
import src.config as config

model = load_model(config.MODEL_PATH)

sample = pd.DataFrame([{
    "island": "Torgersen",
    "culmen_length_mm": 39.1,
    "culmen_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "MALE"
}])

prediction = model.predict(sample)

print("Predicted species:", prediction)
