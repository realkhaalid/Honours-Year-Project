import numpy as np
import torch
import math


def split_heads(
    embeddings,
    num_heads
):
    batch_size, num_patches, embedding_dim = (
        embeddings.shape
    )

    if embedding_dim % num_heads != 0:
        raise ValueError(
            "embedding_dim must be divisible "
            "by num_heads."
        )

    head_dim = (
        embedding_dim // num_heads
    )

    heads = embeddings.reshape(
        batch_size,
        num_patches,
        num_heads,
        head_dim
    )

    heads = heads.permute(
        0,2,1,3
    )

    return heads

def combine_heads(heads):
    batch_size, num_heads, num_patches, head_dim = heads.shape

    heads = heads.permute(
        0, 2, 1, 3
    )

    combined = heads.reshape(
        batch_size, num_patches, num_heads * head_dim
    )

    return combined

def linear_projection(input, W, b):
    return input @ W + b

def calc_attention(
    query,
    key,
    value
):
    """
    Calculates scaled dot-product attention.
    """

    head_dim = query.shape[-1]

    attention_scores = (
        query
        @ key.transpose(-2, -1)
    )

    attention_scores = (
        attention_scores
        / math.sqrt(head_dim)
    )

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1
    )

    attention_output = (
        attention_weights
        @ value
    )

    return (
        attention_output,
        attention_weights
    )

def residual_connection(input, output):
    rc_output = input + output
    
    return rc_output

def layer_norm(input, gamma, beta, eps=1e-5):
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (input - mean) / torch.sqrt(var + eps)
    ln_output = gamma * x_normalized + beta
    return ln_output

def apply_activation(input, activation):
    """
    Applies the selected activation function.
    """

    if activation == "relu":
        return torch.relu(input)

    if activation == "gelu":
        return 0.5 * input * (
            1.0
            + torch.erf(
                input / math.sqrt(2.0)
            )
        )

    if activation == "silu":
        return input * torch.sigmoid(input)

    raise ValueError(
        f"Unsupported activation function: "
        f"{activation}"
    )

def forward_pass(
    input,
    weights,
    biases,
    activation="gelu"
):
    """
    Passes input through a configurable feed-forward
    network.

    All layers except the final layer use the selected
    activation function.
    """

    output = input

    for layer_index in range(
        len(weights)
    ):
        output = (
            output
            @ weights[layer_index]
            + biases[layer_index]
        )

        if layer_index < len(weights) - 1:
            output = apply_activation(
                output,
                activation
            )

    return output