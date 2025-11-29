import numpy as np
import io
import librosa

from pydub import AudioSegment
from src.audio_augmentor import AudioAugmentor

class FeatureExtractor():

    def __init__(self, sr=22050, chunk_duration=1.5, overlaps=[0.0, 0.25, 0.5, 0.75], n_mels=128, n_fft=1024, hop_length=256):
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.overlaps = overlaps
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length # Hop length for STFT (Short-Time Fourier Transform)
        self.n_frames = self.compute_n_frames()
        self.n_channels = 3  # log-mel + delta + delta-delta

        # Create audio augmentor
        self.augmentor = AudioAugmentor(sr=self.sr)

    # Ensures the number of frames in the given logmel is consistent with n_frames.
    def adjust_frames(self, logmel):
        _, current_frames = logmel.shape
    
        if current_frames < self.n_frames:
            # Pad with minimum value (or 0 dB)
            pad_width = self.n_frames - current_frames
            logmel = np.pad(logmel, ((0,0), (0, pad_width)), mode='constant', constant_values=logmel.min())
        elif current_frames > self.n_frames:
            # Truncate extra frames
            logmel = logmel[:, :self.n_frames]
        
        return logmel

    # Loads the audio file, converts to mono, and trims silence
    def load_audio(self, path):
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True) # Load only mono channel of audio (reduces complexity) 
            y, _ = librosa.effects.trim(y)  # Trim leading/trailing silence
            return y
        except Exception as e:
            print(f"Could not load {path}: {e}")
            return None
    
    # Loads the audio file form file storage, converts to mono, and trims silience
    def load_audio_from_filestorage(self, file_storage):
        try:
            data = file_storage.read()
            file_storage.seek(0)  # Reset pointer

            # Load with pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(data))

            # Convert to mono and target sr
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.sr)

            # Get raw samples as numpy array
            y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

            # Normalize audio range
            y /= np.iinfo(audio_segment.array_type).max

            # Trim silence
            y, _ = librosa.effects.trim(y)

            return y
        except Exception as e:
            print(f"Could not load uploaded audio: {e}")
            return None

    def get_audio_chunks(self, y):

        print("\tChunking audio...")

        chunk_len = int(self.chunk_duration * self.sr)

        chunks = []
        n_samples = len(y)

        for offset in self.overlaps:
            hop_len = int(chunk_len * (1 - offset)) # Step size between chunks

            if n_samples < chunk_len:
                # Pad short audio so at least one chunk is created
                padded = np.pad(y, (0, chunk_len - n_samples))
                return [padded]
            
            # Loop from start of audio to end - chunk legth (to avoid partial chunk)
            for start in range(0, n_samples - chunk_len + 1, hop_len):
                end = start + chunk_len
                chunk = y[start:end]
                chunks.append(chunk)

        return chunks
    
    def extract_logmel(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize logmel
        log_mel_spec = self.normalize_logmel(log_mel_spec)

        # Adjust frames to the correct number n_frames
        log_mel_spec = self.adjust_frames(log_mel_spec)

        return log_mel_spec

    # Compute the deltas (1st and 2nd derivatives) of the log-mel spectrogram for additional features
    def compute_deltas(self, logmel):
        delta = librosa.feature.delta(logmel)
        delta2 = librosa.feature.delta(logmel, order=2)
        return delta, delta2
    
    def extract_features_from_audio(self, audio):
        chunks = self.get_audio_chunks(audio)
        features = []

        for chunk in chunks:
            
            # Augment each chunk's audio
            chunk = self.augmentor.light_augmentation(chunk)

            # Extract logmel and compute deltas
            logmel = self.extract_logmel(chunk)
            delta, delta2 = self.compute_deltas(logmel)

            # Stack log-mel and deltas along the first dimension (channels)
            stacked = np.stack([logmel, delta, delta2], axis=0) # Shape: (3, n_mels, time_frames)
            features.append(stacked)
        
        print(f"\tExtracted {len(features)} features.")

        return features

    # Computes the number of frames for log-mel spectrogram based on chunk duration and STFT parameters.
    def compute_n_frames(self):
        # Number of samples in one chunk
        samples = int(self.chunk_duration * self.sr)

        # Number of STFT frames
        n_frames = 1 + int((samples - self.n_fft) // self.hop_length)

        return n_frames
    
    def normalize_logmel(self, spec):
        mean = spec.mean(axis=1, keepdims=True)
        std = spec.std(axis=1, keepdims=True) + 1e-6
        return (spec - mean) / std
