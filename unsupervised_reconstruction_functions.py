import numpy as np
import torch
import math

def mask_patches(patches, mask_ratio=0.3):
    batch_size, num_patches, _ = patches.shape
    mask = torch.rand(batch_size, num_patches) < mask_ratio
    masked_patches = patches.clone()
    masked_patches[mask] = 0.0
    return masked_patches, mask

def reconstruction_head(encoder_output, W_reconstruct, b_reconstruct):
    reconstructed_patches = encoder_output @ W_reconstruct + b_reconstruct
    return reconstructed_patches

def hidden_patch_reconstruction_loss(original_patches, reconstructed_patches, mask):
    mask = mask.unsqueeze(-1)
    squared_error = (reconstructed_patches - original_patches) ** 2
    masked_error = squared_error * mask
    loss = masked_error.sum() / (mask.sum() * original_patches.shape[-1]).clamp(min=1)
    return loss