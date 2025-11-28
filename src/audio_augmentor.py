import numpy as np
import librosa
import io
import scipy.signal as sps

from pydub import AudioSegment

class AudioAugmentor:
    def __init__(self, sr):
        self.sr = sr

    def heavy_augmentation(self, audio):
        
        # Preemphasis boosts high frequencies
        audio = librosa.effects.preemphasis(audio)

        # Add noise to simulate microphone noise
        noise_level = np.random.uniform(0.0, 0.005)
        audio = audio + np.random.randn(len(audio)) * noise_level

        # Add pink noise to simulate room / background noise
        if np.random.rand() < 0.5:
            pink = self.generate_pink_noise(len(audio))
            scale = np.random.uniform(0.001, 0.01)
            audio = audio + pink * scale

        # Shift pitch to prevent model from memorizing pitches
        pitch_shift = np.random.uniform(-1, 1)  # semitones
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=pitch_shift)

        # Stretch time to make model invarient to small differences in playback
        stretch = np.random.uniform(0.95, 1.05)
        if stretch != 1:
            audio = librosa.effects.time_stretch(audio, rate=stretch)

        # Randomly alter EQ to simulate different speakers and mics
        audio = self.random_eq(audio)

        # Add reverb to simulate different acoustic spaces
        if np.random.rand() < 0.4:
            audio = self.add_reverb(audio)

        # Scale volume randomly to ensure model ignores volume differences
        volume = np.random.uniform(0.8, 1.2)
        audio = audio * volume

        # Shift time either left or right to prevent model from memorizing exact alignments
        shift = np.random.randint(-2000, 2000)
        audio = np.roll(audio, shift)

        # Randomly mask parts of the audio to force model to examine global patterns
        if np.random.rand() < 0.3:
            k = np.random.randint(int(0.01*self.sr), int(0.05*self.sr))
            start = np.random.randint(0, len(audio)-k)
            audio[start:start+k] = 0

        # Randomly clip loud sounds to simulate clipping from phone microphones
        if np.random.rand() < 0.4:
            audio = np.clip(audio, -0.9, 0.9)

        # Apply codec compression to simulate lossy audio
        if np.random.rand() < 0.25:
            audio = self.apply_codec_compression(audio)

        # Apply soft knee compression to simulate phone microphones
        if np.random.rand() < 0.3:
            audio = self.soft_knee_compress(audio)

        # Simulate the audio being played out of a device
        if np.random.rand() < 0.4:
            audio = self.simulate_speaker_playback(audio)

        return audio

    def light_augmentation(self, audio):
        # --- 1. Preemphasis (boost high frequencies) ---
        audio = librosa.effects.preemphasis(audio)

        # --- 2. Add light white noise ---
        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.0, 0.005)
            audio = audio + np.random.randn(len(audio)) * noise_level

        # --- 3. Mild pitch shift (±0.25 semitones) ---
        pitch_shift = np.random.uniform(-0.25, 0.25)
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=pitch_shift)

        # --- 4. Slight time stretch (0.98–1.02) ---
        stretch = np.random.uniform(0.98, 1.02)
        if stretch != 1:
            audio = librosa.effects.time_stretch(audio, rate=stretch)

        # --- 5. Light volume scaling (0.8–1.2) ---
        volume = np.random.uniform(0.8, 1.2)
        audio = audio * volume

        # --- 6. Minor time shift (±0.01–0.02 sec) ---
        shift = int(np.random.uniform(-0.02*self.sr, 0.02*self.sr))
        audio = np.roll(audio, shift)

        # --- 7. Optional mild EQ (low/med/high) ---
        if np.random.rand() < 0.3:
            audio = self.random_eq(audio, gain_range=(0.9, 1.1))

        # --- 8. Optional mild reverb ---
        if np.random.rand() < 0.2:
            audio = self.add_reverb(audio, decay=0.1)

        return audio

    def generate_pink_noise(self, n):
        uneven = n % 2
        X = np.random.randn(n // 2 + 1 + uneven) + 1j * np.random.randn(n // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)[::-1]
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        return y / np.max(np.abs(y))

    def random_eq(self, audio, gain_range=(0.9, 1.1)):
        band = np.random.choice(["low", "mid", "high"])
        gain = np.random.uniform(*gain_range)

        if band == "low":
            b, a = sps.butter(3, 300 / (self.sr/2), btype='low')
        elif band == "mid":
            b, a = sps.butter(3, [300/(self.sr/2), 3000/(self.sr/2)], btype='band')
        else:
            b, a = sps.butter(3, 3000 / (self.sr/2), btype='high')

        filtered = sps.lfilter(b, a, audio)
        return audio + (filtered - audio) * gain  # blend lightly

    def add_reverb(self, audio, decay=0.1):
        ir = np.exp(-np.linspace(0, 1, int(0.03*self.sr)) / decay)
        ir = ir / np.max(np.abs(ir))
        return np.convolve(audio, ir, mode='same')
    
    # Applies compression by saving audio as compressed mp3 and reading back 
    def apply_codec_compression(self, audio, bitrate="32k"):
        
        audio_int16 = (audio * 32767).astype(np.int16)
        seg = AudioSegment(audio_int16.tobytes(),
                           frame_rate=self.sr,
                           sample_width=2,
                           channels=1)

        mp3_buffer = io.BytesIO()
        seg.export(mp3_buffer, format="mp3", bitrate=bitrate)

        mp3_buffer.seek(0)
        decoded = AudioSegment.from_mp3(mp3_buffer)
        raw = np.array(decoded.get_array_of_samples()).astype(np.float32) / 32767.0
        return raw

    # Applies soft knee compression. Gradually compresses as audio reaches threshold.
    # Harshly compresses once audio exceeds threshold
    def soft_knee_compress(self, audio, threshold=0.4, ratio=6.0):
        out = np.copy(audio)
        mask = np.abs(out) > threshold
        out[mask] = np.sign(out[mask]) * (
            threshold + (np.abs(out[mask]) - threshold) / ratio
        )
        return out

    def simulate_speaker_playback(self, audio):
        # Low-pass filter (phone speakers lose high frequencies)
        cutoff = np.random.uniform(2500, 4500) # harsh cutoff is fairly realistic
        b, a = sps.butter(4, cutoff / (self.sr/2), btype='low')
        audio = sps.lfilter(b, a, audio)

        # Add tiny resonant peak to simulate speaker cone resonance
        freq = np.random.uniform(400, 1200)
        Q = np.random.uniform(2, 6)
        b, a = sps.iirpeak(freq / (self.sr/2), Q)
        audio = sps.lfilter(b, a, audio)

        # Add mild distortion (phone speakers distort naturally)
        audio = np.tanh(audio * np.random.uniform(1.5, 3.0))

        return audio