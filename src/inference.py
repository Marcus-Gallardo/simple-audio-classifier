import os
import json
import numpy as np

from tensorflow.keras.models import load_model as keras_load
from src.feature_extractor import FeatureExtractor
from src.utils import get_model_path

# Returns the model and label map
def load_model(model_name):
    model_path = os.path.join(get_model_path(), model_name, "model.keras") 

    if not os.path.exists(model_path):
        print(f"Could not find model at {model_path}.")
        return None, None

    meta_path = os.path.join(get_model_path(), model_name, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"Could not find model metadata at {model_path}.")
        return None, None

    print("Loading model...")
    model = keras_load(model_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Ensure label_map is present and casting int keys properly
    if "label_map" not in meta:
        raise KeyError("metadata.json is missling label_map")    

    label_map = {k: int(v) for k, v in meta["label_map"].items()}

    return model, label_map

# Runs a single audio file through the given model and returns the
# predicted class and the probabilities for each class.
def predict_audio(model, label_map, fileobj):

    if not model or not label_map:
        raise ValueError("Could not load model or label map")

    fe = FeatureExtractor()

    audio = fe.load_audio_from_filestorage(fileobj)

    if audio is None:
        raise ValueError("Could not load audio")

    # Extract features (list of chunks)
    features = fe.extract_features_from_audio(audio)
    X = np.array(features, dtype=np.float32)

    # Fix channel order: (samples, mels, frames, channels)
    X = np.transpose(X, (0, 2, 3, 1))

    print("Feature shape:", X.shape)

    # Make prediction for each chunk
    raw_probs = model.predict(X)

    # Average across chunks
    avg_probs = raw_probs.mean(axis=0)

    # Map for decoding labels
    inv_label_map = {v: k for k, v in label_map.items()}

    # Get highest probability class
    pred_idx = int(np.argmax(avg_probs))
    pred_label = inv_label_map[pred_idx]

    # Build labeled probability dictionary
    labeled_probs = {
        inv_label_map[i]: float(avg_probs[i])
        for i in range(len(avg_probs))
    }

    return pred_label, labeled_probs