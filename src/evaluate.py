import sys
import os

from tensorflow.keras.models import load_model
from src.load_dataset import load_dataset
from src.utils import get_model_path
from tensorflow.keras.utils import to_categorical


def evaluate(dataset_name, run_name):
    print("Loading dataset...")
    X, y_int, label_map = load_dataset(dataset_name)
    y = to_categorical(y_int, num_classes=len(label_map))

    model_path = os.path.join(get_model_path(), run_name, "model.keras") 

    if not model_path:
        print(f"Could not find model at {model_path}")
        return

    print("Loading model...")
    model = load_model(model_path)

    print("Evaluating...")
    _, acc = model.evaluate(X, y)
    print("Overall accuracy:", acc)

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
