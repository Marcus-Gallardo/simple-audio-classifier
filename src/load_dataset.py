import numpy as np
import json

from src.utils import get_dataset_path

def load_dataset(name):

    dataset_path = get_dataset_path()

    npz = np.load(f"{dataset_path}/{name}/X_compressed.npz")
    X = npz["arr_0"]

    # Features were saved as float16 to minmize size on disk
    # Convert them back to float32 for CNN
    X = X.astype("float32")

    y_int = np.load(f"{dataset_path}/{name}/y_int.npy")

    with open(f"{dataset_path}/{name}/label_map.json") as f:
        label_map = json.load(f)

    return X, y_int, label_map