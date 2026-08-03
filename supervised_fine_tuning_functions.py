import numpy as np
import torch
import math

def average_pooling(encoder_output):
    pooled_output = encoder_output.mean(dim=1)
    return pooled_output

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