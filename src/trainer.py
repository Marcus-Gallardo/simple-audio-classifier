import os
import json
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.feature_extractor import FeatureExtractor
from src.utils import get_project_root, get_model_path
from src.load_dataset import load_dataset

class Trainer:
    def __init__(self, feature_extractor: FeatureExtractor, model, data_dir):
        self.feature_extractor = feature_extractor
        self.model = model
        self.data_dir = data_dir

    # Load audio files, extract features, and assign labels.
    # def load_data(self):
    #     # Variables to hold features and labels
    #     X = []
    #     y = []

    #     # Dictionary to encode string labels to integers
    #     label_map = {}

    #     # Iterator for string labels
    #     current_label = 0

    #     # Loop through each artist (label)
    #     for dirname in os.listdir(self.data_dir):
    #         artist_dir = os.path.join(self.data_dir, dirname)
        
    #         # Continue if not a directory
    #         if not os.path.isdir(artist_dir):
    #             continue
            
    #         # Create metadata path
    #         meta_path = os.path.join(artist_dir, "metadata.json")

    #         # If there's no metadata, skip this artist
    #         if not os.path.exists(meta_path):
    #             print(f"Skipping {artist_dir}: no metadata.json")
    #             continue

    #         # Read metadata
    #         with open(meta_path, "r", encoding="utf-8") as f:
    #             meta = json.load(f)

    #         # Get songs from metadata
    #         songs = meta.get("songs", {})

    #         # Continue if the metadata has no songs
    #         if not songs:
    #             print(f"No songs listed in metadata for {artist_dir}")
    #             continue

    #         for song_label, filename in songs.items():
    #             filepath = os.path.join(artist_dir, filename)
    #             if not os.path.exists(filepath):
    #                 print(f"Missing file {filepath}, skipping")
    #                 continue

    #             # Map the song label to an integer ID
    #             if song_label not in label_map:
    #                 label_map[song_label] = current_label
    #                 current_label += 1
                
    #             # Load audio and extract features
    #             audio = self.feature_extractor.load_audio(filepath)
    #             if audio is None:
    #                 print(f"Failed to load audio from {filepath}, skipping.")
    #                 continue

    #             # Get audio chunks
    #             features = self.feature_extractor.extract_features_from_audio(audio)

    #             # Add features to dataset
    #             X.extend(features)

    #             # Add appropriate number of integer labels for this audio file
    #             y.extend([label_map[song_label]] * len(features))
            
    #     if len(X) == 0:
    #         # This should raise an error if no data was loaded
    #         raise RuntimeError("No data found - check metadata.json and data_dir")
        

    #     X = np.array(X)
    #     y = np.array(y)

    #     # Convert labels to one-hot encoding
    #     y_onehot = to_categorical(np.array(y))

    #     return X, y_onehot, label_map
    
    def train(self, dataset_name, run_name, test_size=0.2, val_size=0.1, batch_size=32, epochs=50):
        # Load data
        X, y, label_map = load_dataset(dataset_name)

        # You can't stratify the split on one-hot labels
        # [[0, 1, 0], [1, 0, 0]] → [1, 0]
        y_int = np.argmax(y, axis=1)

        # Split train/test
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y_int, test_size=test_size, stratify=y_int, random_state=42
        )

        # Transpose so that channels is last (correct format for Conv2D)
        # (samples, channels, n_mels, n_frames) transposed to (samples, n_mels, n_frames, channels)
        X_train = np.transpose(X_train, (0, 2, 3, 1))
        X_test = np.transpose(X_test, (0, 2, 3, 1))
        
        print(f"X_train after transpose: {X_train}")
        print(f"Shape of X_test after transpose: {X_test}")

        # Split train/validation
        X_train, X_val, y_train_final, y_val_final = train_test_split(
        X_train, y_train_int, test_size=val_size, stratify=y_train_int, random_state=42
        )   

        # Convert back to one-hot
        num_classes = len(label_map)
        y_train = to_categorical(y_train_final, num_classes=num_classes)
        y_val = to_categorical(y_val_final, num_classes=num_classes)
        y_test = to_categorical(y_test_int, num_classes=num_classes)

        print(f"num_classes: {num_classes}")

        print(self.model.summary())

        # Define early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop],
            shuffle=True
        )
        
        # Evaluate
        _, test_acc = self.model.evaluate(X_test, y_test, verbose=1)

        # Define input shape as shape of first training sample
        input_shape = X.shape[1:]

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "run_name": run_name,
            "test_accuracy": float(test_acc),
            "epochs_trained": len(history.history["loss"]),
            "num_classes": num_classes,
            "input_shape": input_shape,
        }

        self.save_model(run_name, metadata, history)

        return self.model
    
    # Saves the trained model to a file
    def save_model(self, run_name, metadata, history):

        # Ensure directory for saving models exists
        run_dir = os.path.join(get_project_root(), get_model_path(), run_name)
        os.makedirs(run_dir, exist_ok=True)

        save_path = os.path.join(run_dir, "model.keras")
        self.model.save(save_path)

        # Save metadata
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training history
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history.history, f, indent=2)

        print(f"Model and metadata saved to {run_dir}")