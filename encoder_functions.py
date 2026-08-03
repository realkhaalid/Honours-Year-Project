import numpy as np
import torch
import math

def split_heads(embeddings, num_heads=4):
    batch_size, num_patches, embedding_dim = embeddings.shape
    head_dim = embedding_dim // num_heads
    heads = embeddings.reshape(batch_size, num_patches, num_heads, head_dim)
    heads = heads.permute(0, 2, 1, 3)
    return heads

def combine_heads(heads):
    batch_size, num_heads, num_patches, head_dim = heads.shape
    heads = heads.permute(0, 2, 1, 3)
    combined = heads.reshape(batch_size, num_patches, num_heads * head_dim)
    return combined

def linear_projection(input, W, b):
    return input @ W + b

def calc_attention(query, key, value, embedding_dim=128, num_heads=4):
    head_dim = embedding_dim // num_heads
    attention_scores = query @ key.transpose(-2, -1)
    attention_scores = attention_scores / math.sqrt(head_dim)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = attention_weights @ value
    return attention_output, attention_weights

def residual_connection(input, output):
    rc_output = input + output
    return rc_output

def layer_norm(input, gamma, beta, eps=1e-5):
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (input - mean) / torch.sqrt(var + eps)
    ln_output = gamma * x_normalized + beta
    return ln_output

def forward_pass(input, w1, b1, w2, b2):
    hidden = input @ w1 + b1
    hidden = torch.relu(hidden)
    ff_output = hidden @ w2 + b2
    return ff_output