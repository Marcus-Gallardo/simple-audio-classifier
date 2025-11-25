import os
import numpy as np
import librosa
import librosa.effects

class FeatureExtractor():

    def __init__(self, sr=22050, chunk_duration=3.0, overlap=0.5, n_mels=128, n_fft=1024, hop_length=512):
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length # Hop length for STFT (Short-Time Fourier Transform)
        self.n_frames = self.compute_n_frames()
        self.n_channels = 3  # log-mel + delta + delta-delta

    # Ensures the number of frames in the given logmel is consistent with n_frames.
    def adjust_frames(self, logmel):
        n_mels, current_frames = logmel.shape
    
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
            print(f"[ERROR] Could not load {path}: {e}")
            return None
    
    def get_audio_chunks(self, y):
        chunk_len = int(self.chunk_duration * self.sr)
        hop_len = int(chunk_len * (1 - self.overlap)) # Step size between chunks

        chunks = []
        n_samples = len(y)

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

        # Adjust frames to the correct number n_frames
        log_mel_spec = self.adjust_frames(log_mel_spec)

        return log_mel_spec

    # Compute the deltas (1st and 2nd derivatives) of the log-mel spectrogram for additional features
    def compute_deltas(self, logmel):
        delta = librosa.feature.delta(logmel)
        delta2 = librosa.feature.delta(logmel, order=2)
        return delta, delta2
    
    # Augment audio by adding random noise and pitch shifting. Helps prevent CNN overfitting.
    def augment(self, audio,
                noise_level_range=(0.0, 0.005),
                pitch_shift_range=(-1, 1),
                time_stretch_range=(0.95, 1.05),
                volume_scale_range=(0.8, 1.2)):
       
        # Gaussian noise
        noise_level = np.random.uniform(*noise_level_range) # Asterisk unpacks the tuple
        audio = audio + np.random.randn(len(audio)) * noise_level

        # Pitch shift
        pitch_shift = np.random.uniform(*pitch_shift_range)
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=pitch_shift)

        # Time stretch
        time_stretch = np.random.uniform(*time_stretch_range)
        if time_stretch != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=time_stretch)

        # Volume scaling
        volume_scale = np.random.uniform(*volume_scale_range)
        audio = audio * volume_scale

        return audio

    # Extracts features from audio. Computes log-mel spectrogram + deltas, with optional augmentation.
    def extract_features_from_audio(self, audio, augment=True):
        chunks = self.get_audio_chunks(audio)
        features = []

        for chunk in chunks:
            
            if augment:
                chunk = self.augment(chunk)

            logmel = self.extract_logmel(chunk)
            delta, delta2 = self.compute_deltas(logmel)

            # Stack log-mel and deltas along the first dimension (channels)
            stacked = np.stack([logmel, delta, delta2], axis=0) # Shape: (3, n_mels, time_frames)
            features.append(stacked)
        
        return features

    # Computes the number of frames for log-mel spectrogram based on chunk duration and STFT parameters.
    def compute_n_frames(self):
        # Number of samples in one chunk
        samples = int(self.chunk_duration * self.sr)

        # Number of STFT frames
        n_frames = 1 + int((samples - self.n_fft) // self.hop_length)

        return n_frames