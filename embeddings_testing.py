import numpy as np
import torch
import math
from generate_datasets_testing import load_and_process_supervised_dataset, load_and_process_unsupervised_dataset, plot_log_mel_spectrogram

def check_dataset_shapes(dataset):
    print("Dataset type:", type(dataset))
    print("Number of samples:", len(dataset))
    print("Torch tensor type:", type(dataset[0]))
    print("Torch tensor dtype:", dataset[0].dtype)
    print("Torch tensor shape:", dataset[0].shape)

def convert_to_patches(spectrogram, patch_size=10):
    n_freq_bins, n_time_frames = spectrogram.shape
    n_patches = n_time_frames // patch_size
    usable_time_frames = n_patches * patch_size
    spectrogram = spectrogram[:, :usable_time_frames]
    patches = spectrogram.reshape(n_freq_bins, n_patches, patch_size)
    patches = patches.permute(1, 0, 2)
    patches = patches.reshape(n_patches, n_freq_bins * patch_size)
    return patches

def check_patch_shapes(patches):
    print("Patches type:", type(patches))
    print("Number of patches:", len(patches))
    print("Torch tensor type:", type(patches[0]))
    print("Torch tensor full patch shape:", patches.shape)
    print("Torch tensor dtype:", patches[0].dtype)
    print("Torch tensor shape:", patches[0].shape)

def convert_to_embeddings(patches, embedding_dim=128):
    patches = patches.unsqueeze(0)
    patch_dim = patches.shape[-1]
    W_embedding = torch.randn(patch_dim, embedding_dim) * 0.01
    W_embedding.requires_grad_()
    b_embedding = torch.zeros(embedding_dim)
    b_embedding.requires_grad_()
    embeddings = patches @ W_embedding + b_embedding
    return embeddings, W_embedding, b_embedding

def check_embedding_shapes(embeddings, W_embedding, b_embedding):
    print("Embeddings shape:", embeddings.shape)
    print("Weight shape:", W_embedding.shape)
    print("Bias shape:", b_embedding.shape)

def positional_encoding(num_patches, embedding_dim):
    pos_encoding = torch.zeros(num_patches, embedding_dim)
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
    torch.arange(0, embedding_dim, 2, dtype=torch.float32)
    * (-math.log(10000.0) / embedding_dim)
    )
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    pos_encoding = pos_encoding.unsqueeze(0)
    return pos_encoding

def add_positional_encoding(embeddings):
    num_patches, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    pos_encoding = pos_encoding.to(embeddings.device)
    embeddings_with_positional_information = embeddings + pos_encoding
    return embeddings_with_positional_information

def check_positional_encoding(pos_encoding, embeddings_with_positional_information, embeddings):
    print("Positional encoding shape:", pos_encoding.shape)
    print("Positional encoding dtype:", pos_encoding.dtype)
    print("Positional encoding sample values:", pos_encoding[0, :5, :5])
    print("Embeddings with positional information shape:", embeddings_with_positional_information.shape)
    print("Embeddings with positional information dtype:", embeddings_with_positional_information.dtype)
    print("Embeddings sample values:", embeddings[0, :5, :5])
    print("Embeddings with positional information sample values:", embeddings_with_positional_information[0, :5, :5])
   
if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels = load_and_process_supervised_dataset(supervised_dataset_path)
    
    print(f"Unsupervised dataset processed with {len(unsupervised_log_mel_specs)} log-mel spectrograms")
    print(f"Supervised dataset processed with {len(supervised_log_mel_specs)} log-mel spectrograms and {len(supervised_labels)} labels")
    plot_log_mel_spectrogram(unsupervised_log_mel_specs[0])

    check_dataset_shapes(unsupervised_log_mel_specs)

    sample_spectrogram = unsupervised_log_mel_specs[0]
    patches = convert_to_patches(sample_spectrogram)
    check_patch_shapes(patches)

    embeddings, W_embedding, b_embedding = convert_to_embeddings(patches)
    check_embedding_shapes(embeddings, W_embedding, b_embedding)

    num_patches, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    embeddings_with_positional_information = add_positional_encoding(embeddings)
    check_positional_encoding(pos_encoding, embeddings_with_positional_information, embeddings)