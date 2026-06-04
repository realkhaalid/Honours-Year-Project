import numpy as np
import torch
import math
from generate_datasets_testing import load_and_process_supervised_dataset, load_and_process_unsupervised_dataset
from embeddings_testing import check_positional_encoding, convert_to_patches, add_positional_encoding, check_embedding_shapes, convert_to_embeddings, positional_encoding

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

def init_multi_head_output_params(embedding_dim=128):
    W_output = torch.randn(embedding_dim, embedding_dim) * 0.01
    W_output.requires_grad_()

    b_output = torch.zeros(embedding_dim)
    b_output.requires_grad_()

    return W_output, b_output

def init_attention_params(embedding_dim=128):
    W_query = torch.randn(embedding_dim, embedding_dim) * 0.01
    W_query.requires_grad_()
    b_query = torch.zeros(embedding_dim)
    b_query.requires_grad_()

    W_key = torch.randn(embedding_dim, embedding_dim) * 0.01
    W_key.requires_grad_()
    b_key = torch.zeros(embedding_dim)
    b_key.requires_grad_()

    W_value = torch.randn(embedding_dim, embedding_dim) * 0.01
    W_value.requires_grad_()
    b_value = torch.zeros(embedding_dim)
    b_value.requires_grad_()

    return W_query, b_query, W_key, b_key, W_value, b_value

def linear_projection(input, W, b):
    return input @ W + b

def calc_attention(query, key, value, embedding_dim=128, num_heads=4):
    head_dim = embedding_dim // num_heads
    attention_scores = query @ key.transpose(-2, -1)
    attention_scores = attention_scores / math.sqrt(head_dim)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = attention_weights @ value
    return attention_output, attention_weights
    

def risidual_connection(input, output):
    rc_output = input + output
    return rc_output

def layer_norm(input, gamma, beta, eps=1e-5):
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (input - mean) / torch.sqrt(var + eps)
    ln_output = gamma * x_normalized + beta
    return ln_output

def init_layer_norm_params(embedding_dim):
    gamma = torch.ones(embedding_dim)
    gamma.requires_grad_()

    beta = torch.zeros(embedding_dim)
    beta.requires_grad_()

    return gamma, beta

def init_forward_pass_params(embedding_dim=128, hidden_dim=512):
    w1 = torch.randn(embedding_dim, hidden_dim) * 0.01
    w1.requires_grad_()
    b1 = torch.zeros(hidden_dim)
    b1.requires_grad_()

    w2 = torch.randn(hidden_dim, embedding_dim) * 0.01
    w2.requires_grad_()
    b2 = torch.zeros(embedding_dim)
    b2.requires_grad_()

    return w1, b1, w2, b2

def forward_pass(input, w1, b1, w2, b2):
    hidden = input @ w1 + b1
    hidden = torch.relu(hidden)
    ff_output = hidden @ w2 + b2
    return ff_output

if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels = load_and_process_supervised_dataset(supervised_dataset_path)
    
    sample_spectrogram = unsupervised_log_mel_specs[0]
    patches = convert_to_patches(sample_spectrogram)

    embeddings, W_embedding, b_embedding = convert_to_embeddings(patches)
    check_embedding_shapes(embeddings, W_embedding, b_embedding)

    batch_size, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    embeddings_with_positional_information = add_positional_encoding(embeddings)
    check_positional_encoding(pos_encoding, embeddings_with_positional_information, embeddings)

    W_query, b_query, W_key, b_key, W_value, b_value = init_attention_params()
    W_output, b_output = init_multi_head_output_params()

    query = linear_projection(embeddings_with_positional_information, W_query, b_query)
    key = linear_projection(embeddings_with_positional_information, W_key, b_key)
    value = linear_projection(embeddings_with_positional_information, W_value, b_value)
    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)

    query_heads = split_heads(query)
    key_heads = split_heads(key)
    value_heads = split_heads(value)
    print("Query heads shape:", query_heads.shape)
    print("Key heads shape:", key_heads.shape)
    print("Value heads shape:", value_heads.shape)

    attention_output, attention_weights = calc_attention(query_heads, key_heads, value_heads, embedding_dim=128, num_heads=4)
    print("Attention output shape:", attention_output.shape)

    combined_attention = combine_heads(attention_output)
    print("Combined attention shape:", combined_attention.shape)

    final_projection = linear_projection(combined_attention, W_output, b_output)
    print("Final projection shape:", final_projection.shape)
    print("Attention weights shape:", attention_weights.shape)

    rc_output = risidual_connection(embeddings_with_positional_information, final_projection)
    print("Output shape after residual connection:", rc_output.shape)
    print("Sample embeddings values:", final_projection[0, :5, :5])
    print("Sample rc_output values:", rc_output[0, :5, :5])

    gamma, beta = init_layer_norm_params(embedding_dim=128)
    ln_output = layer_norm(rc_output, gamma, beta)
    print("Layer norm output shape:", ln_output.shape)
    print("Sample layer norm output values:", ln_output[0, :5, :5])

    w1, b1, w2, b2 = init_forward_pass_params(embedding_dim=128, hidden_dim=512)
    ff_output = forward_pass(ln_output, w1, b1, w2, b2)
    print("FForward pass output shape:", ff_output.shape)
    print("Sample forward pass output values:", ff_output[0, :5, :5])

    ff_risidual_output = risidual_connection(ln_output, ff_output)
    print("Output shape after feed forward residual connection:", ff_risidual_output.shape)
    print("Sample feed forward residual output values:", ff_risidual_output[0, :5, :5])

    gamma2, beta2 = init_layer_norm_params(embedding_dim=128)
    encoder_output = layer_norm(ff_risidual_output, gamma2, beta2)
    print("Encoder output shape:", encoder_output.shape)
    print("Sample encoder output values:", encoder_output[0, :5, :5])

def encoder_layer_output(input):
    W_query, b_query, W_key, b_key, W_value, b_value = init_attention_params()
    W_output, b_output = init_multi_head_output_params()

    query = linear_projection(input, W_query, b_query)
    key = linear_projection(input, W_key, b_key)
    value = linear_projection(input, W_value, b_value)

    query_heads = split_heads(query)
    key_heads = split_heads(key)
    value_heads = split_heads(value)

    attention_output, attention_weights = calc_attention(query_heads, key_heads, value_heads, embedding_dim=128, num_heads=4)
    combined_attention = combine_heads(attention_output)
    final_projection = linear_projection(combined_attention, W_output, b_output)
    rc_output = risidual_connection(input, final_projection)
    
    gamma, beta = init_layer_norm_params(embedding_dim=128)
    ln_output = layer_norm(rc_output, gamma, beta)

    w1, b1, w2, b2 = init_forward_pass_params(embedding_dim=128, hidden_dim=512)
    ff_output = forward_pass(ln_output, w1, b1, w2, b2)

    ff_risidual_output = risidual_connection(ln_output, ff_output)

    gamma2, beta2 = init_layer_norm_params(embedding_dim=128)
    encoder_output = layer_norm(ff_risidual_output, gamma2, beta2)

    return encoder_output

if __name__ == "__main__":
    print("################################################################")
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels = load_and_process_supervised_dataset(supervised_dataset_path)
    
    sample_spectrogram = unsupervised_log_mel_specs[0]
    patches = convert_to_patches(sample_spectrogram)

    embeddings, W_embedding, b_embedding = convert_to_embeddings(patches)
    check_embedding_shapes(embeddings, W_embedding, b_embedding)

    batch_size, num_patches, embedding_dim = embeddings.shape
    pos_encoding = positional_encoding(num_patches, embedding_dim)
    embeddings_with_positional_information = add_positional_encoding(embeddings)
    check_positional_encoding(pos_encoding, embeddings_with_positional_information, embeddings)

    encoder_output = encoder_layer_output(embeddings_with_positional_information)
    print("Encoder output shape:", encoder_output.shape)
    print("Sample encoder output values:", encoder_output[0, :5, :5])