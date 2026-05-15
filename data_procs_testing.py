import librosa as lib
import numpy as np
import matplotlib.pyplot as plt

FRAMESIZE = 1024
HOPLENGTH = 512

def load_audio(file_path):
    audio, sr = lib.load(file_path, sr=None)
    return audio, sr


def compute_log_mel_spectrogram(audio_path):
    audio, sr = load_audio(audio_path)
    mel_spec = lib.feature.melspectrogram(y=audio, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)
    log_mel_spec = lib.power_to_db(mel_spec)
    return log_mel_spec

if __name__ == "__main__":
    audio_path = "archive/toms/Tom Sample 1.wav"
    audio, sr = load_audio(audio_path)
    log_mel_spec = compute_log_mel_spectrogram(audio_path)
    
    print(f"Audio loaded with sample rate: {sr} and length: {len(audio)} samples")

    # Plot the log-mel spectrogram
    plt.figure(figsize=(10, 4))
    lib.display.specshow(log_mel_spec, sr=22050, hop_length=HOPLENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    plt.show()