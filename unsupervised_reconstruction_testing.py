import numpy as np
import torch
import math

from generate_datasets_testing import load_and_process_supervised_dataset, load_and_process_unsupervised_dataset
from embeddings_testing import convert_to_patches, add_positional_encoding, convert_to_embeddings, check_positional_encoding, check_embedding_shapes, positional_encoding
from encoder_testing import encoder_layer_output

def mask_patches(patches, mask_ratio=0.3):
    batch_size, num_patches, patch_dim = patches.shape
    mask = torch.rand(batch_size, num_patches) < mask_ratio
    masked_patches = patches.clone()
    masked_patches[mask] = 0.0
    return masked_patches, mask

def compare_original_and_masked_patches(original_patches, masked_patches, mask):
    print("Original patches shape:", original_patches.shape)
    print("Masked patches shape:", masked_patches.shape)
    print("Mask shape:", mask.shape)

    print("Sample original patch values (first 5 patches):", original_patches[0, :5, :5])
    print("Sample masked patch values (first 5 patches):", masked_patches[0, :5, :5])

    num_masked = mask.sum().item()
    print(f"Number of masked patches: {num_masked} out of {mask.numel()} total patches")

    if num_masked > 0:
        print("Sample original patch values (first 5 masked patches):")
        print(original_patches[mask][:5])
        print("Sample masked patch values (first 5 masked patches):")
        print(masked_patches[mask][:5])
    else:
        print("No patches were masked.")

def init_reconstruction_params(embedding_dim=128, patch_dim=1280):
    W_reconstruct = torch.randn(embedding_dim, patch_dim) * 0.01
    W_reconstruct.requires_grad_()

    b_reconstruct = torch.zeros(patch_dim)
    b_reconstruct.requires_grad_()

    return W_reconstruct, b_reconstruct

def reconstruction_head(encoder_output, W_reconstruct, b_reconstruct):
    reconstructed_patches = encoder_output @ W_reconstruct + b_reconstruct
    return reconstructed_patches

if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"
    
    unsupervised_data = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_data, labels = load_and_process_supervised_dataset(supervised_dataset_path)

    print("Unsupervised dataset shape:", unsupervised_data.shape)
    print("Supervised dataset shape:", supervised_data.shape)
    print("Labels shape:", labels.shape)

    sample_spectrogram = unsupervised_data[0]
    patches = convert_to_patches(sample_spectrogram)
    masked_patches, mask = mask_patches(patches, mask_ratio=0.3)
    compare_original_and_masked_patches(patches, masked_patches, mask)
    
    embeddings, W_embedding, b_embedding = convert_to_embeddings(masked_patches)
    check_embedding_shapes(embeddings, W_embedding, b_embedding)

    batch_size, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    embeddings_with_positional_information = add_positional_encoding(embeddings)
    check_positional_encoding(pos_encoding, embeddings_with_positional_information, embeddings)

    encoder_output = encoder_layer_output(embeddings_with_positional_information)
    print("Encoder output shape:", encoder_output.shape)
    print("Sample encoder output values:", encoder_output[0, :5, :5])

    W_reconstruct, b_reconstruct = init_reconstruction_params(embedding_dim=128, patch_dim=1280)
    reconstructed_patches = reconstruction_head(encoder_output, W_reconstruct, b_reconstruct)
    print("Reconstructed patches shape:", reconstructed_patches.shape)
    print("Sample reconstructed patch values:", reconstructed_patches[0, :5, :5])
    print("Sample original patch values for comparison:", patches[0, :5, :5])

    