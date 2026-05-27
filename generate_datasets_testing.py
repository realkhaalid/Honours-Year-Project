import librosa as lib
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

FRAMESIZE = 1024
HOPLENGTH = 512

def load_audio(file_path):
    audio, sr = lib.load(file_path, sr=None)
    return audio, sr


def compute_log_mel_spectrogram(audio_path):
    audio, sr = load_audio(audio_path)
    mel_spec = lib.feature.melspectrogram(y=audio, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH, n_mels=128)
    log_mel_spec = lib.power_to_db(mel_spec)
    return log_mel_spec

def plot_log_mel_spectrogram(log_mel_spec, sr=44100):
    log_mel_spec = log_mel_spec.detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    lib.display.specshow(log_mel_spec, sr=sr, hop_length=HOPLENGTH, x_axis='time', y_axis='mel', fmax=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def save_log_mel_spectrogram(log_mel_spec, file_path, sr=44100):
    plt.figure(figsize=(10, 4))
    lib.display.specshow(log_mel_spec, sr=sr, hop_length=HOPLENGTH, x_axis='time', y_axis='mel', fmax=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def load_and_process_unsupervised_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    log_mel_specs = []

    for audio_file in dataset_path.rglob("*.wav"):
        log_mel_spec = compute_log_mel_spectrogram(audio_file)
        log_mel_specs.append(log_mel_spec)

    spectrogram = np.array(log_mel_specs, dtype=np.float32)
    tensor = torch.from_numpy(spectrogram)    

    return tensor


def load_and_process_supervised_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    log_mel_specs = []
    labels = []

    for audio_file in dataset_path.rglob("*.wav"):
        log_mel_spec = compute_log_mel_spectrogram(audio_file)
        log_mel_specs.append(log_mel_spec)

        label = audio_file.parent.name
        labels.append(label)

    spectrogram = np.array(log_mel_specs, dtype=np.float32)
    tensor = torch.from_numpy(spectrogram)

    return tensor, np.array(labels)

def check_folder_structure(dataset_path):
    folder = Path(dataset_path)

    for file_path in folder.rglob("*"):
        if file_path.is_file():
            print(file_path.name)

if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"
    
    check_folder_structure(unsupervised_dataset_path)
    check_folder_structure(supervised_dataset_path)

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels = load_and_process_supervised_dataset(supervised_dataset_path)
    
    print(f"Unsupervised dataset processed with {len(unsupervised_log_mel_specs)} log-mel spectrograms")
    print(f"Supervised dataset processed with {len(supervised_log_mel_specs)} log-mel spectrograms and {len(supervised_labels)} labels")
