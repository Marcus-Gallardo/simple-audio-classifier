import os
import json
import numpy as np
from src.feature_extractor import FeatureExtractor
from src.utils import get_raw_audio_path, get_dataset_path
from tensorflow.keras.utils import to_categorical

# Load audio files, extract features, and assign labels.
def save_dataset(dataset_name):

    print("Saving dataset...")

    fe = FeatureExtractor()

    # Variables to hold features and labels
    X = []
    y = []

    # Dictionary to encode string labels to integers
    label_map = {}

    # Iterator for string labels
    current_label = 0

    raw_audio_dir = get_raw_audio_path()

    # Loop through each artist (label)
    for dirname in os.listdir(raw_audio_dir):
        artist_dir = os.path.join(raw_audio_dir, dirname)
    
        # Continue if not a directory
        if not os.path.isdir(artist_dir):
            continue
        
        # Create metadata path
        meta_path = os.path.join(artist_dir, "metadata.json")

        # If there's no metadata, skip this artist
        if not os.path.exists(meta_path):
            print(f"Skipping {artist_dir}: no metadata.json")
            continue

        # Read metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Get songs from metadata
        songs = meta.get("songs", {})

        # Continue if the metadata has no songs
        if not songs:
            print(f"No songs listed in metadata for {artist_dir}")
            continue

        # Get songs from metadata
        for song_label, filename in songs.items():
            filepath = os.path.join(artist_dir, filename)
            if not os.path.exists(filepath):
                print(f"Missing file {filepath}, skipping")
                continue

            # Map the song label to an integer ID
            if song_label not in label_map:
                label_map[song_label] = current_label
                current_label += 1
            
            # Load audio and extract features
            audio = fe.load_audio(filepath)
            if audio is None:
                print(f"Failed to load audio from {filepath}, skipping.")
                continue

            # Get audio chunks
            features = fe.extract_features_from_audio(audio)

            # Add features to dataset
            X.extend(features)

            # Add appropriate number of integer labels for this audio file
            y.extend([label_map[song_label]] * len(features))
        
    if len(X) == 0:
        # This should raise an error if no data was loaded
        raise RuntimeError("No data found - check metadata.json and data_dir")
    

    X = np.array(X)
    y = np.array(y)

    # Convert labels to one-hot encoding
    y_int = to_categorical(np.array(y))

    # Define path where this dataset will be saved
    dataset_path = f"{get_dataset_path()}/{dataset_name}"

    # Create path if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    np.save(f"{dataset_path}/X.npy", X)
    np.save(f"{dataset_path}/y_int.npy", y_int)

    with open(f"{dataset_path}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("Dataset saved!")
    print("X shape:", X.shape)
    print("y shape:", y_int.shape)
    print("Classes:", len(label_map))

if __name__ == "__main__":
    save_dataset("Ten Clairo Songs")