from src import config
from src.data import load_data
from src.model import build_model
from src.train import train
from src.evaluate import evaluate
from src.utils import save_model

# Load data
X_train, X_test, y_train, y_test = load_data(
    config.DATA_PATH,
    0.2,
    42
)

# Build model
model = build_model()

# Train
model = train(model, X_train, y_train)

# Evaluate
acc, report = evaluate(model, X_test, y_test)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Accuracy:", acc)
print("\nReport:\n", report)

# Save model
save_model(model, config.MODEL_PATH)
