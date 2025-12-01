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
downloader = YouTubeDownloader()

songs = ["We fell in love in october", "blessings kettama remix"]

downloader.download_songs(songs)

# artists = ["Clairo", "Rush", "Juice WRLD"]

# downloader.download_artists(artists, max_per_artist=15)

# dataset_name = "Clairo, Rush, Juice WRLD"

# save_dataset(dataset_name=dataset_name)

# Initialize feature extractor
# fe = FeatureExtractor()

# # Load dataset
# _,  _, label_map = load_dataset(dataset_name)

# # Calculate input shape for the model
# input_shape = (fe.n_mels, fe.n_frames, fe.n_channels)

# # Build model
# model = build_cnn_v2(input_shape=input_shape, num_classes=len(label_map))

# # Initialize trainer
# trainer = Trainer(feature_extractor=fe, model=model, data_dir=get_raw_audio_path())

# # Train model
# trainer.train(dataset_name=dataset_name, model_name="run 3")