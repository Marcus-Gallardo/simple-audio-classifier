# import numpy as np
# import matplotlib.pyplot as plt
# import time

import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "12"

from src.feature_extractor import FeatureExtractor
from src.trainer import Trainer
from src.utils import get_raw_audio_path
from src.load_dataset import load_dataset
from src.model_builder import build_cnn_v2
from src.save_dataset import save_dataset
from tensorflow.keras.models import load_model
from src.youtube_downloader import YouTubeDownloader

# from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Initialize YouTube downloader
# downloader = YouTubeDownloader()

# artists = ["Clairo", "Rush", "Juice WRLD"]

# downloader.download_artists(artists, max_per_artist=15)

dataset_name = "Clairo, Rush, Juice WRLD"

# save_dataset(dataset_name=dataset_name)

# Initialize feature extractor
fe = FeatureExtractor()

# Load dataset
_,  _, label_map = load_dataset(dataset_name)

# Calculate input shape for the model
input_shape = (fe.n_mels, fe.n_frames, fe.n_channels)

# Build model
model = build_cnn_v2(input_shape=input_shape, num_classes=len(label_map))

# Initialize trainer
trainer = Trainer(feature_extractor=fe, model=model, data_dir=get_raw_audio_path())

# Train model
trainer.train(dataset_name=dataset_name, model_name="run 3")

# # Get number of songs (number of possible outputs for the model)
# num_classes = downloader.get_num_songs()

# # Calculate input shape for the model
# input_shape = (fe.n_mels, fe.n_frames, fe.n_channels)

# print("Building Model...")
# model = build_mfcc_cnn(input_shape, num_classes)

# # Get path to raw audio data for trainer
# data_dir = get_raw_audio_path()

# Initialize trainer
# trainer = Trainer(feature_extractor=fe, model=model, data_dir=data_dir)

# print("Training Model...")

# # Train model
# history, X_test, y_test, label_map = trainer.train(
#     test_size=0.2,
#     val_size=0.1,
#     batch_size=32,
#     epochs=50,
# )

# print("Saving Model...")
# trainer.save_model(model_name=f"model_{str(time.time())}")

# print("Evaluating Model...")

# # Evaluate model
# y_pred = model.predict(X_test)

# # Convert predictions from onehot to integer class ([0, 1, 0, 0] turns into class index 1)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_test_classes = np.argmax(y_test, axis=1)

# # Invert the label map to get string labels from integers
# inv_label_map = {v: k for k, v in label_map.items()}

# # Convert integers back to string labels for report
# y_pred_labels = [inv_label_map[i] for i in y_pred_classes]
# y_test_labels = [inv_label_map[i] for i in y_test_classes]

# # Get classification report
# print(classification_report(y_test_labels, y_pred_labels))

# # Display confusion matrix
# ConfusionMatrixDisplay.from_predictions(y_test_labels, y_pred_labels)
# plt.xticks(rotation=45)
# plt.show()