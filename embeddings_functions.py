import numpy as np
import torch
import math

def positional_encoding(
    num_patches,
    embedding_dim,
    device,
    dtype
):
    """
    Creates sinusoidal temporal positional encoding.
    """

    pos_encoding = torch.zeros(
        num_patches,
        embedding_dim,
        device=device,
        dtype=dtype
    )

    position = torch.arange(
        num_patches,
        device=device,
        dtype=dtype
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            embedding_dim,
            2,
            device=device,
            dtype=dtype
        )
        * (
            -math.log(10000.0)
            / embedding_dim
        )
    )

    pos_encoding[:, 0::2] = torch.sin(
        position * div_term
    )

    pos_encoding[:, 1::2] = torch.cos(
        position * div_term[
            :pos_encoding[:, 1::2].shape[1]
        ]
    )

    return pos_encoding.unsqueeze(0)

def add_positional_encoding(
    embeddings
):
    """
    Adds temporal positional information.
    """

    _, num_patches, embedding_dim = (
        embeddings.shape
    )

    pos_encoding = positional_encoding(
        num_patches,
        embedding_dim,
        embeddings.device,
        embeddings.dtype
    )

    return embeddings + pos_encoding