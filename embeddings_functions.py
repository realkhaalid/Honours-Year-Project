import numpy as np
import torch
import math

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
    _, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    embeddings_with_positional_information = embeddings + pos_encoding
    return embeddings_with_positional_information