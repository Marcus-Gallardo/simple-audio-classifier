import os
import numpy as np

from tensorflow.keras.models import load_model
from src.feature_extractor import FeatureExtractor
from src.utils import get_dataset_path, get_model_path


def load_label_map(dataset_name):
    """Loads the label_map.json for the dataset the model trained on."""
    import json
    label_map_path = os.path.join(get_dataset_path(), dataset_name, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Invert for decoding
    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map

# Runs a single audio file through the given model and returns the prediction
def predict_audio(model_path, audio_path, dataset_name):

    print(f"\nLoading model: {model_path}")
    model = load_model(model_path)

    print(f"Loading audio: {audio_path}")
    fe = FeatureExtractor()

    audio = fe.load_audio(audio_path)
    if audio is None:
        raise ValueError("Could not load audio")

    # Extract features (list of chunks)
    features = fe.extract_features_from_audio(audio)
    X = np.array(features)

    # Fix channel order: (samples, mels, frames, channels)
    X = np.transpose(X, (0, 2, 3, 1))

    print("Feature shape:", X.shape)

    # Make prediction for each chunk
    raw_probs = model.predict(X)

    # Average across chunks
    avg_probs = raw_probs.mean(axis=0)

    # Map for decoding labels
    inv_label_map = load_label_map(dataset_name)

    # Get highest probability class
    pred_idx = int(np.argmax(avg_probs))
    pred_label = inv_label_map[pred_idx]

    # Build labeled probability dictionary
    labeled_probs = {
        inv_label_map[i]: float(avg_probs[i])
        for i in range(len(avg_probs))
    }

    return pred_label, avg_probs, labeled_probs 


if __name__ == "__main__":
    dataset_name = "Clairo, Rush, Juice WRLD"
    model_name = "run 3"

    model_path = os.path.join(get_model_path(), model_name, "model.keras")
    audio_path = r"C:\Users\mgall\Coding Projects\simple-audio-classifier\data\processed_samples\bags.wav"

    pred_label, _, labeled_probs = predict_audio(model_path, audio_path, dataset_name)

    print(f"Model prediction: {pred_label}")
    print()
    print(f"All prediction probabilities:")
    print()

    for label in labeled_probs.keys():
        print(f"{label}: {round(labeled_probs[label], 3)}")
    print()
