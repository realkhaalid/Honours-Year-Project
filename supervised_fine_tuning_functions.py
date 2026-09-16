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

def cross_entropy_loss(
    logits,
    labels
):
    """
    Calculates numerically stable cross-entropy
    loss directly from logits.
    """

    batch_size = logits.shape[0]

    correct_class_logits = logits[
        torch.arange(
            batch_size,
            device=logits.device
        ),
        labels
    ]

    log_sum_exp = torch.logsumexp(
        logits,
        dim=-1
    )

    loss = (
        log_sum_exp
        - correct_class_logits
    ).mean()

    return loss