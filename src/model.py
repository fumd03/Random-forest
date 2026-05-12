# Additional imports for the model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_model():

    categorical_features = ["island", "sex"]
    numeric_features = ["culmen_length_mm", "culmen_depth_mm",
                        "flipper_length_mm", "body_mass_g"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return model
