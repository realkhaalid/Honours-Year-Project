import numpy as np
import torch
import math

def create_patch_mask(
    patches,
    mask_ratio=0.3
):
    batch_size, num_patches, _ = (
        patches.shape
    )

    number_to_mask = max(
        1,
        round(
            num_patches
            * mask_ratio
        )
    )

    random_values = torch.rand(
        batch_size,
        num_patches,
        device=patches.device
    )

    mask_indices = torch.topk(
        random_values,
        number_to_mask,
        dim=1
    ).indices

    mask = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=patches.device
    )

    mask.scatter_(
        1,
        mask_indices,
        True
    )

    return mask

def reconstruction_head(encoder_output, W_reconstruct, b_reconstruct):
    reconstructed_patches = encoder_output @ W_reconstruct + b_reconstruct
    return reconstructed_patches

def hidden_patch_reconstruction_loss(
    original_patches,
    reconstructed_patches,
    mask
):
    """
    Calculates MSE only over masked spectrogram
    patches.
    """

    squared_error = (
        reconstructed_patches
        - original_patches
    ) ** 2

    masked_error = squared_error[
        mask
    ]

    if masked_error.numel() == 0:
        return squared_error.mean() * 0.0

    return masked_error.mean()