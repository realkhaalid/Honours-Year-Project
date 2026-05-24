import numpy as np
import torch
from generate_datasets_testing import load_and_process_supervised_dataset, load_and_process_unsupervised_dataset

unsupervised_dataset_path = "archive"
supervised_dataset_path = "archive"

unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
supervised_log_mel_specs, supervised_labels = load_and_process_supervised_dataset(supervised_dataset_path)
    
print(f"Unsupervised dataset processed with {len(unsupervised_log_mel_specs)} log-mel spectrograms")
print(f"Supervised dataset processed with {len(supervised_log_mel_specs)} log-mel spectrograms and {len(supervised_labels)} labels")

def check_dataset_shapes(dataset):
    print("Dataset type:", type(dataset))
    print("Number of samples:", len(dataset))
    print("Torch tensor type:", type(dataset[0]))
    print("Torch tensor dtype:", dataset[0].dtype)
    print("Torch tensor shape:", dataset[0].shape)

check_dataset_shapes(unsupervised_log_mel_specs)

def convert_to_patches(spectrogram, patch_size=10):
    n_freq_bins, n_time_frames = spectrogram.shape
    n_patches = n_time_frames // patch_size
    usable_time_frames = n_patches * patch_size
    spectrogram = spectrogram[:, :usable_time_frames]
    


