import os
import json
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.utils import get_raw_audio_path, get_dataset_path
from tensorflow.keras.utils import to_categorical

class DatasetManager:
    def __init__(self):
        pass

    # Loads audio files, extract features, assign labels, and
    # saves with the given dataset name.
    def save_dataset(self, dataset_name, only_artists=None, only_songs=None):

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

            if only_artists:
                # Check to see that this artist should be saved to this new dataset
                if meta["artist"] not in only_artists:
                    continue                

            # Get songs from metadata
            songs = meta.get("songs", {})    

            # Continue if the metadata has no songs
            if not songs:
                print(f"No songs listed in metadata for {artist_dir}")
                continue

            # Get songs from metadata
            for song_label, filename in songs.items():

                if only_songs:
                    # Check to see that this song should be saved to this new dataset
                    if song_label not in only_songs:
                        continue

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

                print(f"Extracting features for {song_label}...")

                # Get audio chunks
                features = fe.extract_features_from_audio(audio)

                # Add features to dataset
                X.extend(features)

                # Add appropriate number of integer labels for this audio file
                y.extend([label_map[song_label]] * len(features))

            
        if len(X) == 0:
            # This should raise an error if no data was loaded
            raise RuntimeError("No data found - check metadata.json and data_dir")
        
        # Use to float16 to reduce file size and memory buffer size in RAM
        X = np.array(X, dtype=np.float16)
        
        y = np.array(y)

        # Convert labels to one-hot encoding
        y_int = to_categorical(np.array(y))

        # Define path where this dataset will be saved
        dataset_path = f"{get_dataset_path()}/{dataset_name}"

        # Create path if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        np.savez_compressed(f"{dataset_path}/X_compressed", X)
        np.save(f"{dataset_path}/y_int.npy", y_int)

        with open(f"{dataset_path}/label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        print("Dataset saved!")
        print("X shape:", X.shape)
        print("y shape:", y_int.shape)
        print("Classes:", len(label_map))

    # Loads a saved dataset given the dataset name.
    # If conver_type, will return the features in float32
    # instead of their saved form (float16). 
    def load_dataset(self, name, convert_type=True):

        dataset_path = get_dataset_path()

        npz = np.load(f"{dataset_path}/{name}/X_compressed.npz")
        X = npz["arr_0"]

        if convert_type:
            # Features were saved as float16 to minmize size on disk
            # Convert them back to float32 for CNN
            X = X.astype("float32")

        y_int = np.load(f"{dataset_path}/{name}/y_int.npy")

        with open(f"{dataset_path}/{name}/label_map.json") as f:
            label_map = json.load(f)

        return X, y_int, label_map
    
    def combine_datasets(self, dataset_names, new_name):
    
        print(f"Combining datasets: {dataset_names} into '{new_name}'")

        # Vars to hold accumulated features and labels
        all_X = []
        all_y = []
        global_label_map = {}
        next_label_id = 0

        for name in dataset_names:
            print(f"Loading dataset '{name}'...")
            X, y_int, label_map = self.load_dataset(name, convert_type=False)

            # y_int is one-hot, so convert back to integers
            y_raw = np.argmax(y_int, axis=1)

            # Build reverse map (int to label_name)
            inv_map = {v: k for k, v in label_map.items()}

            # Remap labels to new global ID space
            new_labels = []
            for old_label in y_raw:
                label_name = inv_map[old_label]

                if label_name not in global_label_map:
                    global_label_map[label_name] = next_label_id
                    next_label_id += 1

                new_labels.append(global_label_map[label_name])

            new_labels = np.array(new_labels)

            print(f"\tAdded X: {X.shape}, y: {new_labels.shape}")

            all_X.append(X)
            all_y.append(new_labels)

        # Concatenate everything
        X_comb = np.concatenate(all_X, axis=0)

        y_comb = np.concatenate(all_y, axis=0)
        y_int_comb = to_categorical(y_comb)

        print("Combined dataset shapes:")
        print("\tX:", X_comb.shape)
        print("\ty:", y_int_comb.shape)
        print("\tClasses:", len(global_label_map))

        # Save new combined dataset
        dataset_path = os.path.join(get_dataset_path(), new_name)
        os.makedirs(dataset_path, exist_ok=True)

        print("Saving combined dataset...")

        np.savez_compressed(f"{dataset_path}/X_compressed", X_comb.astype(np.float16))
        np.save(f"{dataset_path}/y_int.npy", y_int_comb)

        with open(f"{dataset_path}/label_map.json", "w") as f:
            json.dump(global_label_map, f, indent=2)

        print(f"Combined dataset saved to '{new_name}'")

    