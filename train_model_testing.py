import torch
from tqdm import tqdm
from embeddings_testing import convert_to_patches
from transformer_architecture_testing import Transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from generate_datasets_testing import (
    load_and_process_supervised_dataset,
    load_and_process_unsupervised_dataset
)

EPOCHS = 5
LEARNING_RATE = 0.001

def split_supervised_dataset(spectrograms, labels):
    # First split: 70% train, 30% temp
    train_specs, temp_specs, train_labels, temp_labels = train_test_split(
        spectrograms,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels.numpy()
    )

    # Second split: split temp into 15% validation and 15% test
    val_specs, test_specs, val_labels, test_labels = train_test_split(
        temp_specs,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels.numpy()
    )

    return train_specs, train_labels, val_specs, val_labels, test_specs, test_labels

def split_unsupervised_dataset(spectrograms):
    train_specs, val_specs = train_test_split(
        spectrograms,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    return train_specs, val_specs

def unsupervised_training(
    model,
    train_specs,
    val_specs,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0

        for i in range(len(train_specs)):
            spectrogram = train_specs[i]
            patches = convert_to_patches(spectrogram)

            loss, reconstructed_patches, mask, attention_weights = model.unsupervised_reconstruction(
                patches
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_specs)

        total_val_loss = 0.0
        
        for i in range(len(val_specs)):
            spectrogram = val_specs[i]
            patches = convert_to_patches(spectrogram)

            with torch.no_grad():
                loss, reconstructed_patches, mask, attention_weights = model.unsupervised_reconstruction(
                    patches
                )

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_specs)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Reconstruction Loss: {avg_train_loss:.4f} | "
            f"Val Reconstruction Loss: {avg_val_loss:.4f}"
        )

def supervised_training(
    model,
    train_specs,
    train_labels,
    val_specs,
    val_labels,
    epochs=5,
    learning_rate=0.001
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_train_loss = 0.0
        train_correct = 0

        for i in range(len(train_specs)):
            spectrogram = train_specs[i]
            label = train_labels[i].unsqueeze(0)

            patches = convert_to_patches(spectrogram)

            loss, logits, predicted_class, probabilities, attention_weights = model.supervised_fine_tuning(
                patches,
                label
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if predicted_class.item() == label.item():
                train_correct += 1

        avg_train_loss = total_train_loss / len(train_specs)
        train_accuracy = train_correct / len(train_specs)

        total_val_loss = 0.0
        val_correct = 0

        for i in range(len(val_specs)):
            spectrogram = val_specs[i]
            label = val_labels[i].unsqueeze(0)

            patches = convert_to_patches(spectrogram)

            with torch.no_grad():
                loss, logits, predicted_class, probabilities, attention_weights = model.supervised_fine_tuning(
                    patches,
                    label
                )

            total_val_loss += loss.item()

            if predicted_class.item() == label.item():
                val_correct += 1

        avg_val_loss = total_val_loss / len(val_specs)
        val_accuracy = val_correct / len(val_specs)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

def test_model(model, test_specs, test_labels, encoding_map):
    y_true = []
    y_pred = []

    total_loss = 0.0

    for i in range(len(test_specs)):
        spectrogram = test_specs[i]
        label = test_labels[i].unsqueeze(0)

        patches = convert_to_patches(spectrogram)

        with torch.no_grad():
            loss, logits, predicted_class, probabilities, attention_weights = model.supervised_fine_tuning(
                patches,
                label
            )

        total_loss += loss.item()

        y_true.append(label.item())
        y_pred.append(predicted_class.item())

    avg_loss = total_loss / len(test_specs)
    accuracy = accuracy_score(y_true, y_pred)

    class_names = [
        encoding_map[i]
        for i in range(len(encoding_map))
    ]

    print("Test Loss:", avg_loss)
    print("Test Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))               

if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    # Load and process datasets
    unsupervised_spectrograms = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_spectrograms, supervised_labels, encoding_map, label_map = load_and_process_supervised_dataset(supervised_dataset_path)

    # Split the unsupervised dataset into training and validation sets
    unsupervised_train_specs, unsupervised_val_specs = split_unsupervised_dataset(unsupervised_spectrograms)

    # Split the supervised dataset into training, validation, and test sets
    supervised_train_specs, train_labels, supervised_val_specs, val_labels, test_specs, test_labels = split_supervised_dataset(
        supervised_spectrograms,
        supervised_labels
    )

    # Initialize the Transformer model
    model = Transformer()
    print("Transformer model initialized.")

    print("Starting unsupervised training...")
    unsupervised_training(
        model,
        unsupervised_train_specs,
        unsupervised_val_specs,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    print("Starting supervised training...")
    supervised_training(
        model,
        supervised_train_specs,
        train_labels,
        supervised_val_specs,
        val_labels,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    print("Testing the model on the test set...")
    test_model(
        model,
        test_specs,
        test_labels,
        encoding_map
    )