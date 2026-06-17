import numpy as np
import torch
import math

from generate_datasets_testing import load_and_process_supervised_dataset, load_and_process_unsupervised_dataset
from embeddings_testing import convert_to_patches, add_positional_encoding, convert_to_embeddings, check_positional_encoding, check_embedding_shapes, positional_encoding
from encoder_testing import encoder_layer_output

def average_pooling(encoder_output):
    pooled_output = encoder_output.mean(dim=1)
    return pooled_output

def init_classifier_params(embedding_dim=128, num_classes=4):
    W_classifier = torch.randn(embedding_dim, num_classes) * 0.01
    W_classifier.requires_grad_()

    b_classifier = torch.zeros(num_classes)
    b_classifier.requires_grad_()

    return W_classifier, b_classifier

def classification_head(pooled_output, W_classifier, b_classifier):
    logits = pooled_output @ W_classifier + b_classifier
    return logits

def predict_class(logits):
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)

    return predicted_class, probabilities

def cross_entropy_loss(logits, labels):
    probabilities = torch.softmax(logits, dim=-1)
    batch_size = logits.shape[0]
    correct_class_probabilities = probabilities[
        torch.arange(batch_size),
        labels
    ]
    loss = -torch.log(correct_class_probabilities + 1e-9).mean()
    return loss

if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels, encoding_map, label_map = load_and_process_supervised_dataset(supervised_dataset_path)

    sample_label = supervised_labels[0]
    print("Label map: ", label_map)
    print("Encoding map: ", encoding_map)
    print("Sample label shape:", sample_label.shape)
    print("Sample label dtype:", sample_label.dtype)
    print("Sample label:", sample_label.item())
    print("Sample class name:", encoding_map[sample_label.item()])
    
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

    pooled_output = average_pooling(encoder_output)
    print("Pooled output shape:", pooled_output.shape)

    W_classifier, b_classifier = init_classifier_params(embedding_dim=128, num_classes=4)
    logits = classification_head(pooled_output, W_classifier, b_classifier)
    print("Logits shape:", logits.shape)
    print("Sample logits values:", logits[0, :5])

    predicted_class, probabilities = predict_class(logits)
    print("Predicted class:", predicted_class)
    print("Predicted class name: ", encoding_map[predicted_class.item()])
    print("Class probabilities:", probabilities)

    loss = cross_entropy_loss(logits, sample_label)
    print("Cross-entropy loss:", loss)