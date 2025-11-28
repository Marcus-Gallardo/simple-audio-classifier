import numpy as np
import json

from src.utils import get_dataset_path

def load_dataset(name):

    dataset_path = get_dataset_path()

    X = np.load(f"{dataset_path}/{name}/X.npy")
    y_int = np.load(f"{dataset_path}/{name}/y_int.npy")

    with open(f"{dataset_path}/{name}/label_map.json") as f:
        label_map = json.load(f)

    return X, y_int, label_map